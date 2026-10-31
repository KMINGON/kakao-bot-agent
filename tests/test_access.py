import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from kakao_bot.runtime import load_pack, lock_pack
from kakao_bot.runtime.action_api import ActionAPI


spec = importlib.util.spec_from_file_location('test_gyodong_access', Path('domain_packs/gyodong/access.py'))
access = importlib.util.module_from_spec(spec)
spec.loader.exec_module(access)
SELF, OTHER = '100000001', '100000002'
MEMBER = {'member_id':'MB-SELF','name':'회원본인','phone':'019-1111-1234','status':'ACTIVE',
          'address':'PRIVATE_MEMBER_ADDRESS','birth_year':1990,'grade':'MEMBER','member_type':'INDIVIDUAL'}
BRIEF = 'TC ID: L-23\ncase metadata\n- id: L-23\n- 상담 인입 일시: 2037-06-30 10:00\n- 인입창구: 조합몰 채팅\n- 인입 계정: 회원 로그인 · 회원본인 · 019-****-1234\n\n'


def state():
    return {'tc_id':'L-23','case_brief':BRIEF + f'주문 {SELF}, {OTHER} 확인해주세요.',
            'messages':[HumanMessage(content='변경하고 환불해 주세요.')],
            'action_observations':{},'action_requests':{},'policy_reads':{'D015':True}}


class Boundary:
    def __init__(self):
        self.calls=[]
        self.order_status='ok'
    async def fetch(self, api_id, parameters):
        self.calls.append((api_id,parameters))
        order=parameters.get('order_no',SELF)
        owner='MB-SELF' if order==SELF else 'MB-OTHER'
        records={
            'API-13':[MEMBER,{**MEMBER,'member_id':'MB-OTHER','phone':'019-2222-5678'}],
            'API-07':{'order_no':order,'member_id':owner,'amount':30000,'status':'PAID',
                      'recipient_name':'PRIVATE_RECIPIENT','shipping_address':'PRIVATE_STREET',
                      'items':[{'sku':'SKU-1'}],'payment_refs':['PAY-OWN']},
            'API-11':[{'order_no':order,'shipment_id':'SHP-OWN','status':'READY','picked_up_at':None,
                       'delivered_location':'PRIVATE_DELIVERY_LOCATION','items':[{'lot_id':'LOT-OWN','sku':'SKU-1'}]}],
            'API-10':{'order_no':SELF,'shipment_id':parameters.get('shipment_id'),'status':'READY',
                      'picked_up_at':None,'items':[{'lot_id':'LOT-OWN','sku':'SKU-1'}]},
            'API-08':[{'order_no':SELF,'member_id':'MB-SELF','shipping_address':'PRIVATE_OWN'},
                      {'order_no':OTHER,'member_id':'MB-OTHER','shipping_address':'PRIVATE_OTHER'}],
            'API-17':{'approval_no':parameters.get('approval_no'),'order_no':SELF,'amount':30000,
                      'payer_name':'PRIVATE_PAYER','card_last4':'1234'},
            'API-19':{'subscription_id':parameters.get('subscription_id'),'member_id':'MB-OTHER'},
            'API-12':MEMBER,
            'API-06':[{'lot_id':'LOT-OWN','member_id':'MB-OTHER','content':'PRIVATE_TICKET','handler':'PRIVATE_STAFF'}],
            'API-05':[{'ticket_id':'T-OWN','member_id':'MB-SELF','content':'PRIVATE_TICKET','handler':'PRIVATE_STAFF'}],
            'API-14':[{**MEMBER,'member_id':'MB-OTHER'}],
            'API-01':{'sku':'SKU-1','name':'공개상품'},
            'API-20':{'code':parameters.get('code')},
        }
        if api_id=='API-07' and self.order_status!='ok':
            return {'api_id':api_id,'parameters':parameters,'status':self.order_status,'http_status':404,'evidence':[]}
        return {'api_id':api_id,'parameters':parameters,'status':'ok','data':records.get(api_id,{}),
                'effect':'read_only','evidence':[{'source_id':api_id}]}


async def get(boundary,s,api_id,parameters):
    result,updates=await access.guarded_get(api_id,parameters,s,boundary.fetch)
    s['action_observations'].update(updates)
    return result


def assert_private_absent(value):
    text=json.dumps(value,ensure_ascii=False,default=str)
    assert 'PRIVATE_' not in text and '019-1111-1234' not in text and '019-2222-5678' not in text


@pytest.mark.asyncio
async def test_other_order_returns_only_mismatch_and_cannot_query_shipping():
    b,s=Boundary(),state()
    result=await get(b,s,'API-07',{'order_no':OTHER})
    assert result['status']=='forbidden'
    assert result['data']=={'order_no':OTHER,'owner_mismatch':True,'handoff_required':True}
    assert 'MB-OTHER' not in json.dumps(result)
    assert_private_absent([result,s['action_observations']])
    assert (await get(b,s,'API-11',{'order_no':OTHER}))['status']=='forbidden'
    assert all(api_id!='API-11' for api_id,_ in b.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize('api_id,parameters',[
    ('API-08',{'member_id':'MB-OTHER'}),('API-12',{'member_id':'MB-OTHER'}),
    ('API-13',{'name':'다른회원'}),('API-15',{'member_id':'MB-OTHER'}),('API-16',{'member_id':'MB-OTHER'}),
    ('API-17',{'approval_no':'PAY-OTHER'}),('API-10',{'shipment_id':'SHP-OTHER'}),
    ('API-03',{'lot_id':'LOT-OTHER'}),('API-06',{'lot_id':'LOT-OTHER'}),
    ('API-36',{'escalation_id':'ESC-OTHER'})])
async def test_indirect_private_queries_are_blocked_before_target_fetch(api_id,parameters):
    b,s=Boundary(),state()
    result=await get(b,s,api_id,parameters)
    assert result['status'] in {'needs_input','forbidden'}
    assert (api_id,parameters) not in b.calls
    assert_private_absent([result,s['action_observations']])


@pytest.mark.asyncio
async def test_same_account_order_payment_and_lot_are_minimized_and_linked():
    b,s=Boundary(),state()
    order=await get(b,s,'API-07',{'order_no':SELF})
    assert order['data']['amount']==30000
    await get(b,s,'API-11',{'order_no':SELF})
    await get(b,s,'API-17',{'approval_no':'PAY-OWN'})
    aggregate=await get(b,s,'API-06',{'lot_id':'LOT-OWN'})
    assert aggregate['data']['ticket_count']==aggregate['data']['other_customer_ticket_count']==1
    assert aggregate['data']['reported_symptom_counts']=={'MOLD':0,'DAMAGE':0,'UNKNOWN':1}
    await get(b,s,'API-05',{'member_id':'MB-SELF'})
    assert_private_absent(s['action_observations'])
    assert sum(api_id=='API-13' for api_id,_ in b.calls)==1


@pytest.mark.parametrize('content,expected',[
    ('수박에 곰팡이가 피어 있음. 환불 요청.', 'MOLD'),
    ('곰팡이 발생 신고 접수', 'MOLD'),
    ('배송 중 파손 신고 접수', 'DAMAGE'),
    ('수박이 파손되어 도착했습니다.', 'DAMAGE'),
    ('곰팡이 없다', 'UNKNOWN'),
    ('곰팡이가 발생하지 않았습니다.', 'UNKNOWN'),
    ('파손 신고에 해당하지 않음.', 'UNKNOWN'),
    ('곰팡이 발생 신고 철회. 검은 흙으로 확인.', 'UNKNOWN'),
    ('곰팡이가 피어 있음처럼 보였으나 포장 인쇄였음.', 'UNKNOWN'),
    ('기록 예시: 파손 신고 접수', 'UNKNOWN'),
    ('곰팡이 발생 시 신고하세요.', 'UNKNOWN'),
    ('곰팡이 발생 신고 예정', 'UNKNOWN'),
    ('곰팡이 발생 빈도', 'UNKNOWN'),
    ('파손 발생 비율', 'UNKNOWN'),
    ('곰팡이 발생 조건 설명', 'UNKNOWN'),
    ('곰팡이 발생 신고를 해도 되나요?', 'UNKNOWN'),
    ('곰팡이 발생 신고를 하지 마세요.', 'UNKNOWN'),
    ('곰팡이가 피어 있다고 상상한다.', 'UNKNOWN'),
    ('곰팡이가 안 생겼습니다.', 'UNKNOWN'),
    ('곰팡이 발생 신고 및 파손 신고', 'UNKNOWN'),
    ('KD-WTM-370624-02: 곰팡이 발생 신고', 'UNKNOWN'),
    ('배송 상태 문의', 'UNKNOWN'),
    ('', 'UNKNOWN'),
])
def test_ticket_symptoms_require_unambiguous_positive_reports(content,expected):
    result=access.aggregate_lot_tickets([{'ticket_id':'T-1','member_id':'MB-OTHER','content':content}],
                                        'LOT-OWN','MB-SELF')
    assert result['reported_symptom_counts']=={name:int(name==expected) for name in ['MOLD','DAMAGE','UNKNOWN']}
    assert result['symptom_basis']=='reported_not_confirmed'


def test_ticket_aggregation_deduplicates_and_keeps_conflicting_duplicates_unknown():
    def row(ident,content):return {'ticket_id':ident,'member_id':'MB-OTHER','content':content}
    rows=[row('T-1','곰팡이 발생 신고'),row('T-1','곰팡이 발생 신고'),row('T-2','파손 신고'),
          row('T-3','곰팡이 발생 신고'),row('T-3','파손 신고')]
    result=access.aggregate_lot_tickets(rows,'LOT-OWN','MB-SELF')
    assert result['ticket_count']==result['other_customer_ticket_count']==3
    assert result['duplicate_ticket_count']==2
    assert result['reported_symptom_counts']=={'MOLD':1,'DAMAGE':1,'UNKNOWN':1}


def test_ticket_aggregation_never_returns_private_text_ids_or_arbitrary_categories():
    private=['PRIVATE-NAME','010-1234-9876','PRIVATE-ADDRESS','PRIVATE-HANDLER','PRIVATE-ID','MB-OTHER','2037-06-30T12:34:56']
    rows=[{'ticket_id':private[4],'member_id':private[5],'created_at':private[6],
           'content':'곰팡이 발생 신고. '+' '.join(private),'handler':private[3]},
          {'ticket_id':'SECOND-ID','member_id':'MB-SELF','category':'DAMAGE','content':'곰팡이 발생 신고'},
          {'ticket_id':'THIRD-ID','member_id':'MB-SELF','category':'PRIVATE-CATEGORY','content':'곰팡이 발생 신고'}]
    result=access.aggregate_lot_tickets(rows,'LOT-OWN','MB-SELF')
    assert result['reported_symptom_counts']=={'MOLD':1,'DAMAGE':1,'UNKNOWN':1}
    assert result['other_customer_ticket_count']==1
    serialized=json.dumps(result,ensure_ascii=False)
    assert all(value not in serialized for value in private+['SECOND-ID','THIRD-ID','PRIVATE-CATEGORY'])
    assert set(result)=={'lot_id','ticket_count','other_customer_ticket_count','duplicate_ticket_count',
                         'reported_symptom_counts','other_customer_reported_symptom_counts','symptom_basis','interpretation','privacy'}


def test_other_customer_symptoms_cannot_be_inferred_from_separate_counts():
    rows = [{'ticket_id': 'own', 'member_id': 'MB-SELF', 'category': 'MOLD'},
            {'ticket_id': 'other', 'member_id': 'MB-OTHER', 'category': 'UNKNOWN'}]
    result = access.aggregate_lot_tickets(rows, 'LOT-OWN', 'MB-SELF')
    assert result['reported_symptom_counts']['MOLD'] == 1
    assert result['other_customer_ticket_count'] == 1
    assert result['other_customer_reported_symptom_counts'] == {'MOLD': 0, 'DAMAGE': 0, 'UNKNOWN': 1}


def test_empty_ticket_aggregation_does_not_assert_no_defect():
    result=access.aggregate_lot_tickets([],'LOT-OWN','MB-SELF')
    assert result['ticket_count']==result['other_customer_ticket_count']==result['duplicate_ticket_count']==0
    assert result['reported_symptom_counts']=={'MOLD':0,'DAMAGE':0,'UNKNOWN':0}
    assert '0건이 이상 없음을 뜻하지 않습니다' in result['interpretation']


@pytest.mark.asyncio
async def test_member_lists_are_filtered_and_referrals_aggregated():
    b,s=Boundary(),state()
    result=await get(b,s,'API-08',{'member_id':'MB-SELF'})
    assert [row['order_no'] for row in result['data']]==[SELF]
    result=await get(b,s,'API-14',{'referred_by':'MB-SELF'})
    assert result['data']=={'referred_member_count':1}
    assert_private_absent(s['action_observations'])


@pytest.mark.asyncio
async def test_subscription_ownership_is_checked_before_disclosure():
    b,s=Boundary(),state()
    assert (await get(b,s,'API-19',{'subscription_id':'SUB-OTHER'}))['status']=='forbidden'
    assert 'MB-OTHER' not in json.dumps(s['action_observations'])


@pytest.mark.asyncio
async def test_account_requires_initial_metadata_and_ignores_later_claims():
    b,s=Boundary(),state()
    s['case_brief']='이름 회원본인이고 전화는 019-1111-1234입니다.'
    s['messages'].append(HumanMessage(content=BRIEF+f'주문 {SELF}'))
    assert (await get(b,s,'API-07',{'order_no':SELF}))['error']=='account_unverified'
    assert not b.calls
    assert (await get(b,s,'API-20',{'code':'SUBAK15'}))['status']=='ok'


@pytest.mark.asyncio
async def test_auth_cache_cannot_cross_tc_and_404_is_not_an_owned_order():
    b,s=Boundary(),state()
    await get(b,s,'API-07',{'order_no':SELF})
    s['tc_id']='L-32'
    assert (await get(b,s,'API-07',{'order_no':SELF}))['error']=='account_unverified'
    b,s=Boundary(),state();b.order_status='failed'
    result=await get(b,s,'API-11',{'order_no':SELF})
    assert result['status']=='failed' and all(api_id!='API-11' for api_id,_ in b.calls)


@pytest.fixture
def tool_pack(tmp_path,monkeypatch):
    shutil.copytree('domain_packs/gyodong',tmp_path/'gyodong',ignore=shutil.ignore_patterns('__pycache__'))
    lock_pack(tmp_path/'gyodong');pack=load_pack(tmp_path,'gyodong')
    monkeypatch.setenv('ACTION_API_WRITES_ENABLED','true')
    return pack


@pytest.mark.asyncio
@pytest.mark.parametrize('api_id,parameters,body',[
    ('API-24',{'order_no':OTHER},{'new_address':'고객 새 주소'}),
    ('API-26',{'order_no':OTHER},{'memo':'변경 요청'}),
    ('API-29',{}, {'order_no':OTHER,'amount':100,'reason':'환불 요청'}),
    ('API-31',{'approval_no':'PAY-OTHER'},{}),
    ('API-32',{'approval_no':'PAY-OTHER'},{'receipt_number':'010-1111-2222'}),
    ('API-33',{'member_id':'MB-OTHER'},{'promo_code':'CARE5000'}),
    ('API-40',{'subscription_id':'SUB-OTHER'},{'kind':'CANCEL'}),
])
async def test_actual_submit_tool_blocks_cross_account_mutations(tool_pack,monkeypatch,api_id,parameters,body):
    b,s=Boundary(),state()
    async def fake(self,operation,**kwargs):
        assert kwargs.get('method','GET')=='GET','forbidden POST reached transport'
        return await b.fetch(operation,kwargs['parameters'])
    monkeypatch.setattr(ActionAPI,'request',fake)
    tool=next(tool for tool in tool_pack.python_tools() if tool.name=='action_submit')
    command=await tool.coroutine(api_id=api_id,parameters=parameters,body=body,policy_source_ids=['D015'],
        policy_reason='고객이 요청한 대상 변경을 확인합니다.',customer_confirmation='변경하고 환불해 주세요.',
        runtime=SimpleNamespace(state=s,tool_call_id='test'))
    result=json.loads(command.update['messages'][0].content)
    assert result['status'] in {'forbidden','needs_input'}
    assert_private_absent(command.update)


@pytest.mark.asyncio
async def test_actual_get_blocks_leak_and_handoff_remains_allowed(tool_pack,monkeypatch):
    b,s=Boundary(),state();post=[]
    async def fake(self,operation,**kwargs):
        if kwargs.get('method')=='POST':
            post.append(operation)
            return {'status':'submitted','data':{'escalation_id':'ESC-OWN','request_status':'SUBMITTED'}}
        return await b.fetch(operation,kwargs['parameters'])
    monkeypatch.setattr(ActionAPI,'request',fake)
    tools={tool.name:tool for tool in tool_pack.python_tools()}
    command=await tools['action_get'].coroutine(api_id='API-07',parameters={'order_no':OTHER},
        runtime=SimpleNamespace(state=s,tool_call_id='get'))
    assert json.loads(command.update['messages'][0].content)['error']=='order_owner_mismatch'
    assert_private_absent(command.update)
    s['action_observations'].update(command.update['action_observations'])
    command=await tools['action_submit'].coroutine(api_id='API-35',parameters={},
        body={'lane':'LANE2_HANDOFF','reason':f'주문 {OTHER}의 계정 접근권 확인 요청'},
        policy_source_ids=['D015'],policy_reason='타 계정 주문 접근 요청은 담당자 확인으로 이관합니다.',
        customer_confirmation='',issue_key=OTHER+':접근권확인',runtime=SimpleNamespace(state=s,tool_call_id='post'))
    assert post==['API-35'] and json.loads(command.update['messages'][0].content)['status']=='submitted'


def test_target_record_uses_order_api_not_shipment_amount(tool_pack):
    spec=importlib.util.spec_from_file_location('guard_tools',tool_pack.root/'tools.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    s={'action_observations':{
        'shipment':{'status':'ok','api_id':'API-11','data':[{'order_no':SELF,'shipment_id':'SHP-OWN','amount':1}]},
        'order':{'status':'ok','api_id':'API-07','data':{'order_no':SELF,'amount':30000}}}}
    assert module._target_record(s,'order_no',SELF)['amount']==30000


@pytest.mark.asyncio
@pytest.mark.parametrize('api_id,parameters,body',[
    ('API-24',{'order_no':SELF},{'new_address':'고객이 제공한 새 주소'}),
    ('API-29',{}, {'order_no':SELF,'amount':30000,'reason':'고객 환불 요청'}),
    ('API-32',{'approval_no':'PAY-OWN'},{'receipt_number':'010-1111-2222'}),
])
async def test_owned_submissions_still_reach_transport(tool_pack,monkeypatch,api_id,parameters,body):
    b,s=Boundary(),state();posts=[]
    s['messages'].append(HumanMessage(content={'API-24':'새 주소로 변경해주세요.',
        'API-29':'전액 환불해주세요.','API-32':'소득공제용으로 발행 요청해주세요.'}[api_id]))
    await get(b,s,'API-07',{'order_no':SELF})
    await get(b,s,'API-11',{'order_no':SELF})
    async def fake(self,operation,**kwargs):
        if kwargs.get('method')=='POST':
            posts.append(operation)
            return {'status':'submitted','data':{'request_id':'OWN-1','request_status':'SUBMITTED'}}
        return await b.fetch(operation,kwargs['parameters'])
    monkeypatch.setattr(ActionAPI,'request',fake)
    tool=next(tool for tool in tool_pack.python_tools() if tool.name=='action_submit')
    command=await tool.coroutine(api_id=api_id,parameters=parameters,body=body,policy_source_ids=['D015'],
        policy_reason='확인된 본인 주문에 대해 고객이 요청한 조치입니다.',customer_confirmation='변경하고 환불해 주세요.',
        runtime=SimpleNamespace(state=s,tool_call_id='submit'))
    assert posts==[api_id] and json.loads(command.update['messages'][0].content)['status']=='submitted'


@pytest.mark.asyncio
@pytest.mark.parametrize('choice,allowed', [
    ('다음 배로 다시 보내주세요. 기다리겠습니다. 오래 포장돼 있으면 상하지 않을까요?', True),
    ('다음 운송편으로 보내주세요.', True),
    ('다음 배로 보내줄 수 있나요?', False),
    ('다음 배로 보내지 마세요.', False),
])
async def test_next_transport_reship_needs_no_date(tool_pack, monkeypatch, choice, allowed):
    boundary, current = Boundary(), state()
    current['messages'] = [HumanMessage(content=choice)]
    await get(boundary, current, 'API-07', {'order_no': SELF})
    await get(boundary, current, 'API-11', {'order_no': SELF})
    posts = []

    async def fake(self, operation, **kwargs):
        if kwargs.get('method') == 'POST':
            posts.append((operation, kwargs['parameters'], kwargs['body']))
            return {'status': 'submitted', 'data': {'request_id': 'RSH-test', 'request_status': 'SUBMITTED'}}
        return await boundary.fetch(operation, kwargs['parameters'])

    monkeypatch.setattr(ActionAPI, 'request', fake)
    submit = next(tool for tool in tool_pack.python_tools() if tool.name == 'action_submit')
    command = await submit.coroutine(api_id='API-28', parameters={'shipment_id': 'SHP-OWN'}, body={},
        policy_source_ids=['D015'], policy_reason='집하 전 결항 배송의 다음 운송편 재출고를 고객이 선택했습니다.',
        runtime=SimpleNamespace(state=current, tool_call_id='reship'))
    result = json.loads(command.update['messages'][0].content)
    assert result['status'] == ('submitted' if allowed else 'needs_input')
    assert posts == ([('API-28', {'shipment_id': 'SHP-OWN'}, {})] if allowed else [])


@pytest.mark.parametrize('refund_order,report_lot,mold,others,status,expected', [
    (SELF, 'LOT-OWN', 1, 1, 'submitted', True),
    (OTHER, 'LOT-OWN', 1, 1, 'submitted', False),
    (SELF, 'LOT-OTHER', 1, 1, 'submitted', False),
    (SELF, 'LOT-OWN', 0, 1, 'submitted', False),
    (SELF, 'LOT-OWN', 1, 0, 'submitted', False),
    (SELF, 'LOT-OWN', 1, 1, 'unknown', False),
])
def test_refund_safety_reminder_uses_only_same_order_report(tool_pack, refund_order, report_lot,
                                                          mold, others, status, expected):
    current = {'action_observations': {
        'shipping': {'api_id': 'API-11', 'status': 'ok', 'data': [
            {'order_no': SELF, 'shipment_id': 'SHP-OWN', 'items': [{'lot_id': 'LOT-OWN'}]}]},
        'reports': {'api_id': 'API-06', 'status': 'ok', 'parameters': {'lot_id': report_lot},
                    'data': {'other_customer_ticket_count': others, 'reported_symptom_counts': {'MOLD': mold},
                             'other_customer_reported_symptom_counts': {'MOLD': min(mold, others)}}},
    }}
    receipt = {'status': status, 'data': {'request_id': 'RFD-test'}}
    result = tool_pack._module()._receipt_guidance('API-29', {'order_no': refund_order}, receipt, current)
    assert ('next_step' in result) is expected
    assert result['data'] == receipt['data']
    assert 'next_step' not in receipt


def test_handoff_receipt_reminds_policy_window_without_claiming_completion(tool_pack):
    sources = {row['id']: row for row in json.loads((tool_pack.root/'data/policies.json').read_text())['sources']}
    receipt = {'status': 'submitted', 'data': {'escalation_id': 'ESC-test'}, 'evidence': [{'source_id': 'API-35'}]}
    result = tool_pack._module()._receipt_guidance('API-35', {}, receipt, {}, sources)
    assert result['data'] == receipt['data']
    assert 'D001' in result['next_step'] and '회신 기한' in result['next_step']
    assert 'D022' in result['next_step']
    excerpt = result['followup_policy']
    lines = sources['D001']['text'].splitlines()
    assert excerpt['text'] == '\n'.join(lines[excerpt['start_line']-1:excerpt['end_line']])
    assert excerpt['text'].startswith('## 접수·회신·승인\n')
    assert lines[excerpt['end_line']].startswith('## ')
    assert '3영업일' in excerpt['text'] and '안내 예시' in excerpt['text']
    assert {row['source_id'] for row in result['evidence']} == {'API-35', 'D001'}
    assert 'next_step' not in receipt
    assert receipt['evidence'] == [{'source_id': 'API-35'}]


@pytest.mark.parametrize('source', [None, {}, {'text': '## 다른 절\n회신 조건 없음'}])
def test_handoff_without_registered_section_does_not_invent_policy(tool_pack, source):
    sources = {'D001': source} if source is not None else {}
    receipt = {'status': 'submitted', 'data': {'escalation_id': 'ESC-test'}}
    result = tool_pack._module()._receipt_guidance('API-35', {}, receipt, {}, sources)
    assert 'followup_policy' not in result and not result.get('evidence')
    assert '3영업일' not in json.dumps(result, ensure_ascii=False)


@pytest.mark.asyncio
async def test_handoff_policy_excerpt_preserves_guard_journal_and_receipt_reuse(tool_pack, monkeypatch):
    current, posts = state(), []
    receipt = {'status': 'submitted', 'data': {'escalation_id': 'ESC-test'}, 'evidence': [{'source_id': 'API-35'}]}
    async def fake(self, operation, **kwargs):
        posts.append((operation, kwargs))
        return receipt
    monkeypatch.setattr(ActionAPI, 'request', fake)
    submit = next(tool for tool in tool_pack.python_tools() if tool.name == 'action_submit')
    args = dict(api_id='API-35', parameters={}, body={'lane': 'LANE2_HANDOFF', 'reason': '잔여 배송 일정 확인'},
        policy_source_ids=['D015'], policy_reason='부분 배송의 잔여 물량 일정을 확인하도록 담당자에게 이관합니다.',
        runtime=SimpleNamespace(state=current, tool_call_id='handoff'), issue_key=SELF+':배송')
    current['policy_reads'] = {}
    blocked = await submit.coroutine(**args)
    assert json.loads(blocked.update['messages'][0].content)['status'] == 'needs_input'
    assert not posts
    current['policy_reads'] = {'D015': True}
    command = await submit.coroutine(**args)
    result = json.loads(command.update['messages'][0].content)
    journal = next(iter(command.update['action_requests'].values()))
    assert result['followup_policy']['source_id'] == 'D001'
    assert journal['result'] == receipt and 'policy_reads' not in command.update
    current['action_requests'].update(command.update['action_requests'])
    reused = json.loads((await submit.coroutine(**args)).update['messages'][0].content)
    assert reused['reused_receipt'] and reused['followup_policy'] == result['followup_policy']
    assert len(posts) == 1 and current['policy_reads'] == {'D015': True}


@pytest.mark.parametrize('other_symptom,expected', [('MOLD', True), ('DAMAGE', False), ('UNKNOWN', False)])
def test_refund_reminder_uses_other_customer_symptom_intersection(tool_pack, other_symptom, expected):
    reports = access.aggregate_lot_tickets([
        {'ticket_id': 'own', 'member_id': 'MB-SELF', 'category': 'MOLD'},
        {'ticket_id': 'other', 'member_id': 'MB-OTHER', 'category': other_symptom},
    ], 'LOT-OWN', 'MB-SELF')
    current = {'action_observations': {
        'shipping': {'api_id': 'API-11', 'status': 'ok', 'data': [
            {'order_no': SELF, 'shipment_id': 'SHP-OWN', 'items': [{'lot_id': 'LOT-OWN'}]}]},
        'reports': {'api_id': 'API-06', 'status': 'ok', 'parameters': {'lot_id': 'LOT-OWN'},
                    'data': reports},
    }}
    receipt = {'status': 'submitted', 'data': {'request_id': 'RFD-test'}}
    result = tool_pack._module()._receipt_guidance('API-29', {'order_no': SELF}, receipt, current)
    assert ('next_step' in result) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize('api_id,parameters,data',[
    ('API-17',{'approval_no':'PAY-OWN'},{'approval_no':'PAY-OWN','order_no':OTHER}),
    ('API-10',{'shipment_id':'SHP-OWN'},{'shipment_id':'SHP-OWN','order_no':OTHER}),
    ('API-03',{'lot_id':'LOT-OWN'},{'lot_id':'LOT-OWN','sku':'WRONG-SKU'}),
])
async def test_linked_results_cannot_switch_even_between_owned_orders(api_id,parameters,data):
    b,s=Boundary(),state()
    await get(b,s,'API-07',{'order_no':SELF});await get(b,s,'API-11',{'order_no':SELF})
    s['action_observations']['other-own']={'api_id':'API-07','status':'ok',
        'data':{'order_no':OTHER,'member_id':'MB-SELF','payment_refs':[]}}
    async def changed(operation,targets):
        if operation==api_id:return {'status':'ok','api_id':operation,'parameters':targets,'data':data}
        return await b.fetch(operation,targets)
    result,_=await access.guarded_get(api_id,parameters,s,changed)
    assert result['error']=='linked_target_mismatch' and 'data' not in result


@pytest.mark.parametrize('new_api,old_api',[('API-28','API-29'),('API-29','API-28')])
@pytest.mark.parametrize('old_status',['submitted','unknown','uncertain','failed'])
@pytest.mark.parametrize('same_order',[True,False])
def test_refund_and_reship_conflict_is_bidirectional_and_order_scoped(tool_pack,new_api,old_api,old_status,same_order):
    spec=importlib.util.spec_from_file_location('conflict_tools',tool_pack.root/'tools.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    s=state()
    s['messages'].append(HumanMessage(content='재배송해주세요.' if new_api=='API-28' else '환불해주세요.'))
    s['action_observations']={
        'order':{'status':'ok','api_id':'API-07','data':{'order_no':SELF,'amount':30000}},
        'shipments':{'status':'ok','api_id':'API-11','data':[
            {'order_no':SELF,'shipment_id':'SHP-OWN'}, {'order_no':OTHER,'shipment_id':'SHP-OTHER'}]}}
    previous_order=SELF if same_order else OTHER
    old_parameters={'shipment_id':'SHP-OWN' if same_order else 'SHP-OTHER'} if old_api=='API-28' else {}
    old_body={'order_no':previous_order,'amount':30000,'reason':'고객 요청'} if old_api=='API-29' else {}
    s['action_requests']={'previous':{'api_id':old_api,'result':{'status':old_status},
        'canonical':json.dumps({'parameters':old_parameters,'body':old_body})}}
    parameters={'shipment_id':'SHP-OWN'} if new_api=='API-28' else {}
    body={'order_no':SELF,'amount':30000,'reason':'고객 요청'} if new_api=='API-29' else {}
    blocked=module.submission_guard(new_api,parameters,body,s,['D015'],'변경하고 환불해 주세요.','고객이 확인한 주문의 처리 요청입니다.')
    assert bool(blocked)==(same_order and old_status in {'submitted','unknown','uncertain'})
    if blocked:assert '철회하는 API가 없으므로' in blocked


@pytest.mark.asyncio
async def test_server_uses_latest_exact_customer_text_even_for_model_paraphrase(tool_pack,monkeypatch):
    b,s=Boundary(),state();calls=[]
    latest='이 주문은 전액 환불해 주세요. 다른 주문은 그대로 두세요.'
    s['messages'].append(HumanMessage(content=latest))
    async def fake(self,operation,**kwargs):
        if kwargs.get('method')=='POST':
            calls.append(kwargs)
            return {'status':'submitted','data':{'request_id':'REFUND-1','request_status':'SUBMITTED'}}
        return await b.fetch(operation,kwargs['parameters'])
    monkeypatch.setattr(ActionAPI,'request',fake)
    tool=next(tool for tool in tool_pack.python_tools() if tool.name=='action_submit')
    args={'api_id':'API-29','parameters':{},'body':{'order_no':SELF,'amount':30000,'reason':'하자'},
          'policy_source_ids':['D015'],'policy_reason':'확인된 본인 주문의 전액 환불을 고객이 요청했습니다.',
          'runtime':SimpleNamespace(state=s,tool_call_id='auto')}
    command=await tool.coroutine(**args)
    journal=next(iter(command.update['action_requests'].values()))
    assert journal['customer_confirmation']==latest and len(calls)==1
    s['action_requests'].update(command.update['action_requests'])
    command=await tool.coroutine(**args)
    assert json.loads(command.update['messages'][0].content)['reused_receipt'] and len(calls)==1
    command=await tool.coroutine(**args,customer_confirmation='고객이 하지 않은 요청 문장')
    assert json.loads(command.update['messages'][0].content)['reused_receipt'] and len(calls)==1
    s['messages'].append(HumanMessage(content='이제 환불하지 마세요.'))
    command=await tool.coroutine(**args,customer_confirmation=latest)
    assert json.loads(command.update['messages'][0].content)['status']=='needs_input' and len(calls)==1


@pytest.mark.parametrize('operation,text,allowed',[
    ('API-29','전액 환불해 주세요.',True),
    ('API-29','환불 요청합니다.',True),
    ('API-29','환불해줘.',True),
    ('API-29','환불이요',True),
    ('API-29','환불해 주지 마세요.',False),
    ('API-29','환불 안 받을게요.',False),
    ('API-29','환불해주세요. 아니요, 취소합니다.',False),
    ('API-29','환불해 달라고 전에 말했지만 지금은 그 요청을 철회합니다.',False),
    ('API-29','주문번호는 100000001이에요.',False),
    ('API-29','환불이 가능한가요?',False),
    ('API-29','환불해주세요. 언제 처리되나요?',True),
    ('API-29','언제 처리되나요? 환불해주세요.',True),
    ('API-29','환불해주세요, 언제 처리되나요?',True),
    ('API-29','환불이 가능한가요? 언제 처리되나요?',False),
    ('API-29','"환불해주세요."라고 하면 바로 접수되나요?',False),
    ('API-29','“환불해주세요.”라는 문구의 뜻이 궁금해요.',False),
    ('API-29','환불해주세요. 언제 처리되나요? 아니요, 철회합니다.',False),
    ('API-29','환불해주세요. 이제 진행하지 않겠습니다.',False),
    ('API-29','환불해주세요. 원하지 않습니다.',False),
    ('API-29','환불해주세요. 이제 하지 않겠습니다.',False),
    ('API-29','환불이 가능한가요? 재배송해주세요.',False),
    ('API-28','환불이 가능한가요? 재배송해주세요.',True),
    ('API-28','다음 배로 다시 보내주세요. 기다리겠습니다. 오래 포장돼 있으면 상하지 않을까요?',True),
    ('API-28','재배송과 환불 둘 다 해주세요. 언제 처리되나요?',False),
    ('API-29','네',False),
    ('API-29','환불 대신 재배송을 선택할게요.',False),
    ('API-28','환불 대신 그 주문을 다시 보내주세요. 재배송을 선택할게요.',True),
    ('API-28','환불하지 말고 재배송해주세요.',True),
    ('API-28','재배송 말고 환불해주세요.',False),
    ('API-28','취소하지 말고 다음 배로 보내주세요. 기다릴게요.',True),
    ('API-28','재배송해주세요. 그만할게요.',False),
    ('API-28','재배송과 환불 둘 다 해주세요.',False),
    ('API-24','배송지를 새 주소로 변경 요청해주세요.',True),
    ('API-23','주문 배송지 주소만 변경해주세요.',False),
    ('API-24','주소 변경하지 말고 영수증만 발행해주세요.',False),
    ('API-32','소득공제용으로 처음 드린 번호로 발행 요청해주세요.',True),
    ('API-33','그 주문의 가격 차액 보상을 요청해요.',True),
    ('API-31','카드 결제를 취소해주세요.',True),
    ('API-40','구독을 취소해주세요.',True),
])
def test_latest_action_choice_overrides_old_consent(tool_pack,operation,text,allowed):
    spec=importlib.util.spec_from_file_location('choice_tools',tool_pack.root/'tools.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    s=state();s['messages']=[HumanMessage(content='전액 환불해 주세요.'),HumanMessage(content=text)]
    body={'kind':'CANCEL'} if operation=='API-40' else {'order_no':SELF,'amount':30000,'reason':'하자'}
    assert (module._current_choice_guard(operation,{},body,s) is None)==allowed


@pytest.mark.asyncio
async def test_model_paraphrase_is_normalized_but_different_current_action_never_posts(tool_pack,monkeypatch):
    b,s=Boundary(),state();posts=[]
    latest='그럼 환불 대신 그 주문을 다시 보내주세요. 재배송을 선택할게요.'
    s['messages'].append(HumanMessage(content=latest))
    await get(b,s,'API-07',{'order_no':SELF});await get(b,s,'API-11',{'order_no':SELF})
    async def fake(self,operation,**kwargs):
        if kwargs.get('method')=='POST':
            posts.append(operation)
            return {'status':'submitted','data':{'request_id':'REPLACE-1','request_status':'SUBMITTED'}}
        return await b.fetch(operation,kwargs['parameters'])
    monkeypatch.setattr(ActionAPI,'request',fake)
    submit=next(tool for tool in tool_pack.python_tools() if tool.name=='action_submit')
    common={'policy_source_ids':['D015'],'policy_reason':'확인된 품질 하자에 대해 고객이 재배송을 선택했습니다.',
            'customer_confirmation':'재배송 신청','runtime':SimpleNamespace(state=s,tool_call_id='paraphrase')}
    denied=await submit.coroutine(api_id='API-29',parameters={},body={'order_no':SELF,'amount':30000,'reason':'하자'},**common)
    assert json.loads(denied.update['messages'][0].content)['status']=='needs_input' and not posts
    accepted=await submit.coroutine(api_id='API-28',parameters={'shipment_id':'SHP-OWN'},body={},**common)
    assert posts==['API-28']
    journal=next(iter(accepted.update['action_requests'].values()))
    assert journal['customer_confirmation']==latest

"use strict";
let state = null;
let working = false;
let draftTimer;
let pendingSave = Promise.resolve();
let quizFailed = false;
const root = document.querySelector("#app");
const labels = {c: "단위 매입가", p: "단위 판매가", v: "잔존가치", b: "추가 부족 비용"};
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const number = value => value === null || value === undefined ? "미확정" : Number(value).toLocaleString("ko-KR", {maximumFractionDigits: 2});
const requestId = () => Array.from(crypto.getRandomValues(new Uint8Array(16)), x=>x.toString(16).padStart(2,'0')).join('');

function error(message) {
  const panel = document.querySelector("#error"); panel.textContent = message; panel.hidden = false;
  setTimeout(() => { panel.hidden = true; }, 6500);
}
async function api(path, data, method = "POST") {
  const response = await fetch(path, {method, headers: {"Content-Type":"application/json", "X-CSRF-Token":state?.csrf || ""}, ...(data === undefined ? {} : {body:JSON.stringify(data)})});
  const content = await response.json();
  if (!response.ok) throw new Error(content.error || "연결에 실패했습니다. 다시 시도해 주세요.");
  return content;
}
async function act(type, fields = {}) {
  if (working) return;
  working = true; clearTimeout(draftTimer);
  try { await pendingSave; const previous = `${state.phase}:${state.trial}`; state = await api("/api/action", {type, requestId: requestId(), ...fields}); render(); if (previous !== `${state.phase}:${state.trial}`) window.scrollTo(0,0); }
  catch (e) {error(e.message);}
  finally {working = false;}
}

function consent() {
  return `<section class="card narrow"><span class="eyebrow">참여 안내와 동의</span><h1>정보를 읽고, 발주량을 결정해 주세요.</h1>
  <p class="intro">문서와 판매 이력을 보고 상품을 얼마나 주문할지 결정하는 연구입니다. 일부 참여자에게는 AI가 정리한 정보가 추가로 제시됩니다.</p>
  <ul><li>예상 시간은 20~30분입니다. 연습 2문항과 본 과제 8문항을 수행합니다.</li><li>제시되는 기업·거래 조건은 실험용입니다. 실제 주문이나 금전 거래는 일어나지 않습니다.</li><li>AI 정보 유무는 무작위로 정해집니다. AI의 제안에는 오류가 있을 수 있습니다.</li><li>매니저 확인은 정해진 답변을 제공하는 실험 기능입니다. 실제 사람에게 메시지를 보내지 않습니다.</li><li>참여는 자발적이며 언제든 중단할 수 있습니다. 중단·삭제 버튼을 누르면 과제 응답은 삭제되고 익명 배정·중단 여부만 남습니다.</li><li>이름·연락처·IP 주소는 저장하지 않습니다. 익명 코드, 동의, 경험, 질문, 답변, 발주량, 확신도와 경과 시간이 기록됩니다.</li></ul>
  <div class="notice"><strong>보상:</strong> ${esc(state.compensation)}<br><strong>문의:</strong> ${esc(state.contact)}</div>
  <p class="muted">잠재적인 불편은 숫자 판단에 따른 피로와 부담입니다. 성적이나 직무 평가와 관계없이 자유롭게 참여를 결정해 주세요.</p>
  <label class="spaced">재고관리 경험<select id="experience"><option value="">선택하세요</option><option value="none">관련 경험 없음</option><option value="course">수업·학습 경험 있음</option><option value="work">실무 경험 있음</option></select></label>
  <label class="check"><input id="adult" type="checkbox">만 19세 이상입니다.</label><label class="check"><input id="agree" type="checkbox">위 내용을 이해했고 자발적으로 참여에 동의합니다.</label>
  <button class="primary" id="consent">동의하고 안내 보기</button><button class="text-button spaced" id="decline">참여하지 않기</button></section>`;
}
function instructions() {
  return `<section class="card narrow"><span class="eyebrow">진행 방법</span><h1>많이 남아도, 부족해도 비용이 듭니다.</h1>
  <p>7일 동안 판매할 상품의 발주량을 정합니다. 판매 후 남은 상품은 반품할 수 있습니다.</p>
  <div class="grid"><div class="notice"><strong>남는 수량 1단위의 비용</strong><br>매입가 − 잔존가치</div><div class="notice"><strong>부족한 수량 1단위의 비용</strong><br>판매가 − 매입가 + 추가 부족 비용</div></div>
  <p><strong>잔존가치</strong>는 환불액에서 처리 수수료를 뺀 금액입니다. 묶음 가격은 묶음당 수량으로 나누어 단위 매입가를 구합니다. 상품·적용 기간·버전이 맞는 문서를 읽어 주세요.</p>
  <p>과거 판매 막대가 주황색이면 품절이 있었습니다. 그날의 판매량은 실제 수요보다 적을 수 있습니다. 미래 수요의 정답은 알려지지 않습니다.</p>
  <p>정보가 부족하거나 충돌하면 매니저에게 확인할 수 있습니다. <strong>한 과제에서 최대 두 번, 한 항목당 한 번</strong> 요청할 수 있으며, 요청당 4점이 총비용에 추가됩니다. 발주를 보류하면 400점의 비용이 듭니다.</p>
  <p>실험 점수는 올바른 정보와 수요분포를 아는 기준 결정에 비해 추가로 발생한 기대비용과 질문 비용입니다. 점수가 낮을수록 좋습니다. 지급 보상과 실험 점수는 모집 안내에 달리 명시되지 않은 한 별개입니다.</p>
  <div class="notice">AI 정보가 보이는 경우에도 최종 판단은 직접 합니다. 제안을 그대로 사용하거나 수정하거나 무시해도 됩니다. 외부 AI·검색을 사용하거나 다른 참여자와 답을 공유하지 말아 주세요.</div>
  <h2 class="spaced">시작 전 확인</h2>${quizFailed ? '<p class="notice warning">안내를 다시 읽고 세 항목을 확인해 주세요.</p>' : ''}
  <form id="quiz" class="quiz"><section><strong>1. 매니저에게 질문하면 어떻게 되나요?</strong><label><input type="radio" name="q1" value="free" required>항상 무료입니다.</label><label><input type="radio" name="q1" value="cost">정해진 비용이 추가됩니다.</label></section>
  <section><strong>2. 정보가 이미 충분한 경우 질문해야 하나요?</strong><label><input type="radio" name="q2" value="mandatory" required>항상 질문해야 합니다.</label><label><input type="radio" name="q2" value="optional">필요한 경우에만 질문합니다.</label></section>
  <section><strong>3. 최종 발주량은 누가 정하나요?</strong><label><input type="radio" name="q3" value="own" required>참여자가 직접 정합니다.</label><label><input type="radio" name="q3" value="auto">시스템이 자동으로 확정합니다.</label></section><button class="primary">확인하고 연습 시작</button></form></section>`;
}
function documents() {
  return state.docs.map(doc => `<article class="document"><h3>${esc(doc.title)}</h3><div class="metadata"><span>${esc(doc.sku)}</span><span>버전 ${esc(doc.version)}</span><span>${esc(doc.period)}</span></div><p>${esc(doc.text)}</p></article>`).join("");
}
function chart() {
  const rows = state.observations, maximum = Math.max(...rows.map(r=>r.sales),1);
  return `<div class="chart" role="img" aria-label="과거 일별 판매량. 주황 막대는 품절 발생 날짜입니다.">${rows.map(r=>`<div class="bar ${r.stockoutHours ? 'censored' : ''}" style="height:${Math.max(3,r.sales/maximum*100)}%" title="${esc(r.date)}: 판매 ${number(r.sales)}${r.stockoutHours ? ' · 품절 있음' : ''}"></div>`).join("")}</div><div class="chart-legend"><span>${esc(rows[0].date)}</span><span>초록: 품절 없음 · 주황: 품절 있음</span><span>${esc(rows.at(-1).date)}</span></div><details><summary>일별 판매 표 보기</summary><div class="scroll-table"><table><thead><tr><th>날짜</th><th>판매량</th><th>품절 시간</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${esc(r.date)}</td><td>${number(r.sales)}</td><td>${number(r.stockoutHours)}</td></tr>`).join("")}</tbody></table></div></details>`;
}
function advice() {
  const a = state.advice;
  if (!a) return "";
  const states = {verified:"확인",conflict:"충돌",unconfirmed:"미확정",unavailable:"확인 불가",candidate:"일부 정보"};
  return `<section class="card advice"><span class="eyebrow">AI 정보 지원</span><h2>AI가 정리한 내용</h2><small>추정이나 문서 해석이 틀릴 수 있습니다. 근거를 확인하고 최종 판단해 주세요.</small>
  <table><thead><tr><th>항목</th><th>값</th><th>상태·근거</th></tr></thead><tbody>${a.parameters.map(p=>`<tr><td>${labels[p.name]}</td><td>${number(p.value)}</td><td>${esc(states[p.state] || p.state)}<small>${esc(state.docs.find(d=>d.id===p.source)?.title || '근거 미확정')}</small></td></tr>`).join("")}</tbody></table>
  ${a.forecast ? `<p>7일 수요 평균 <strong>${number(a.forecast.mean)}</strong><br><small>예측 80% 구간: ${number(a.forecast.low)}~${number(a.forecast.high)}. 정확도가 보장되는 구간은 아닙니다.</small></p>` : '<p>수요를 추정하지 못했습니다.</p>'}
  <p><strong>${a.q === null ? '모수 확인이 필요합니다.' : `제안 발주량 ${number(a.q)}`}</strong></p>
  ${a.missing.length ? `<p class="muted">미확정 항목: ${a.missing.map(s=>labels[s]||'수요').join(', ')}</p>` : ''}
  <button type="button" id="use-fields">AI 모수 값을 입력란에 복사</button>${a.q === null ? '' : '<button type="button" class="spaced" id="use-quantity">제안 발주량을 입력</button>'}</section>`;
}
function task() {
  const draft = state.draft || {};
  return `<div class="layout"><div><section class="card"><span class="eyebrow">${state.practice ? '연습 과제' : '본 과제'}</span><h1>${esc(state.task.sku)}의 7일 발주</h1><p class="muted">판매 기간 ${esc(state.task.period)} · 발주 가능 범위 0~70</p><div class="notice">매니저가 정한 추가 부족 비용은 단위당 <strong>${number(state.managerPolicy)}점</strong>입니다. 모든 금액은 실험 점수 단위이며, 수량은 소수로도 입력할 수 있습니다.</div></section>
  ${advice()}<section class="card"><h2>업무 문서</h2><div class="docs">${documents()}</div></section><section class="card"><h2>최근 35일 판매</h2>${chart()}</section>
  <section class="card"><h2>매니저에게 확인</h2><p class="muted">확인당 4점 · 남은 요청 ${state.remaining}회 · 한 항목당 한 번</p><div class="question-buttons">${['c','p','v'].map(s=>`<button type="button" data-question="${s}" ${state.remaining===0 || state.questions.includes(s) ? 'disabled' : ''}>${labels[s]} 확인</button>`).join('')}</div>${state.questions.length ? `<p class="question-result">확인한 항목: ${state.questions.map(s=>labels[s]).join(', ')}. 답변 문서가 위 목록에 반영됐습니다.</p>` : ''}</section></div>
  <aside class="sticky"><form class="card" id="decision"><h2>나의 판단</h2><p class="muted">근거가 부족하면 해당 항목을 비워 두세요. 추정값을 억지로 입력할 필요는 없습니다.</p>
  ${['c','p','v'].map(s=>`<div class="field-row"><label for="value-${s}">${labels[s]}</label><div class="grid"><input id="value-${s}" type="number" min="0" max="10000" step="any" placeholder="미확정" value="${esc(draft.fields?.[s]?.value ?? '')}"><select id="source-${s}" aria-label="${labels[s]} 근거"><option value="">근거 없음 / 미확정</option>${state.docs.map(d=>`<option value="${esc(d.id)}" ${draft.fields?.[s]?.source===d.id?'selected':''}>${esc(d.title)} · ${esc(d.sku)} · v${d.version}</option>`).join('')}</select></div></div>`).join('')}
  <div class="formula" id="worksheet">모수를 입력하면 단위 과잉·부족 비용을 계산합니다.</div>
  <div class="actions"><label><input type="radio" name="action" value="order" ${draft.action!=='hold'?'checked':''}>발주</label><label><input type="radio" name="action" value="hold" ${draft.action==='hold'?'checked':''}>보류 (400점)</label></div>
  <label class="spaced" for="quantity">최종 발주량<input id="quantity" type="number" min="0" max="70" step="any" placeholder="0~70" value="${esc(draft.q ?? '')}"></label>
  <label class="spaced" for="confidence">이 판단에 얼마나 확신하나요?<select id="confidence" required><option value="">선택하세요</option>${[1,2,3,4,5,6,7].map(n=>`<option value="${n}" ${draft.confidence===n?'selected':''}>${n}${n===1?' — 매우 낮음':n===7?' — 매우 높음':''}</option>`).join('')}</select></label>
  <button class="primary">${state.practice ? '연습 답변 제출' : '판단 확정 · 다음 과제'}</button><small>제출하면 이 과제의 답변을 바꿀 수 없습니다.</small></form></aside></div>`;
}
function gather() {
  const readNumber = id => document.querySelector(id).value.trim()==='' ? null : Number(document.querySelector(id).value);
  return {action:document.querySelector('input[name="action"]:checked').value, q:readNumber('#quantity'), confidence:readNumber('#confidence'), fields:Object.fromEntries(['c','p','v'].map(s=>[s,{value:readNumber('#value-'+s),source:document.querySelector('#source-'+s).value || null}]))};
}
function worksheet() {
  const draft = gather();
  const {c,p,v} = Object.fromEntries(Object.entries(draft.fields).map(([s,f])=>[s,f.value]));
  document.querySelector('#quantity').disabled = draft.action==='hold';
  document.querySelector('#quantity').required = draft.action==='order';
  document.querySelector('#worksheet').textContent = [c,p,v].every(x=>x!==null) ? `남는 수량 1단위 비용: ${number(c-v)}점 · 부족한 수량 1단위 비용: ${number(p-c+state.managerPolicy)}점` : '모수를 입력하면 단위 과잉·부족 비용을 계산합니다.';
}
async function saveDraft() {
  if (state.phase !== 'task' || working) return;
  const payload = {type:'draft', requestId:requestId(),trial:state.trial,draft:gather()};
  pendingSave = pendingSave.then(()=>api('/api/action',payload)).catch(e=>error('임시 저장 실패: '+e.message));
  return pendingSave;
}
function render() {
  document.querySelector('#status').textContent = state.phase==='task' || state.phase==='feedback' ? `${state.practice?'연습':'본 과제'} · ${state.progress} / ${state.total}` : '참여 코드 '+state.id;
  document.querySelector('#contact').textContent = '문의: '+state.contact;
  document.querySelector('#withdraw').hidden = ['declined','withdrawn','consent'].includes(state.phase);
  const banner = state.demo ? '<div class="demo-banner">리허설 모드 · 이 응답은 실제 참여자 자료와 분리됩니다.</div>' : '';
  if (state.phase==='consent') root.innerHTML=banner+consent();
  else if(state.phase==='instructions') root.innerHTML=banner+instructions();
  else if(state.phase==='task') root.innerHTML=banner+task();
  else if(state.phase==='feedback') root.innerHTML=banner+`<section class="card narrow"><span class="eyebrow">연습 피드백</span><h1>연습 답변이 저장됐습니다.</h1><p>완전한 정보를 아는 기준 발주량은 <strong>${number(state.feedback.optimalQ)}</strong>입니다. 내가 정한 수량은 ${number(state.feedback.chosenQ)}입니다.</p><p>이 과제의 추가 기대비용과 질문 비용 합계는 <strong>${number(state.feedback.total)}점</strong>이며, 이 중 질문 비용은 ${number(state.feedback.questionCost)}점입니다.</p><div class="notice">본 과제에서는 정답이나 점수를 즉시 보여 주지 않습니다. AI 제안과 연습 정답이 다를 수 있으며, 최종 판단은 직접 해 주세요.</div><button class="primary" id="next">${state.progress===2?'본 과제 시작':'다음 연습'}</button></section>`;
  else root.innerHTML=`<section class="card narrow complete"><span class="eyebrow">${state.phase==='done'?'참여 완료':'참여 종료'}</span><h1>${state.phase==='done'?'참여해 주셔서 감사합니다.':'참여를 종료했습니다.'}</h1><p>${state.phase==='withdrawn'?'과제 응답이 삭제됐습니다.':state.phase==='declined'?'참여하지 않은 것에 따른 불이익은 없습니다.':'답변이 안전하게 저장됐습니다. 진행자에게 완료 코드를 알려 주세요.'}</p><p class="metric">${esc(state.id)}</p>${state.phase==='done'?'<p>이 실험에서는 AI 지원 유무가 무작위로 배정됐습니다. 과제와 매니저 응답은 통제된 실험 자료였습니다.</p>':''}</section>`;
  bind();
}
function bind() {
  document.querySelector('#consent')?.addEventListener('click',()=>{if(!document.querySelector('#adult').checked||!document.querySelector('#agree').checked)return error('참여 동의와 연령 항목을 확인해 주세요.');act('consent',{adult:true,agreed:true,experience:document.querySelector('#experience').value});});
  document.querySelector('#decline')?.addEventListener('click',()=>act('consent',{agreed:false}));
  document.querySelector('#quiz')?.addEventListener('submit',async e=>{e.preventDefault();quizFailed=true;const data=new FormData(e.target);await act('quiz',{answers:['q1','q2','q3'].map(k=>data.get(k))});});
  document.querySelector('#next')?.addEventListener('click',()=>act('next'));
  document.querySelectorAll('[data-question]').forEach(button=>button.addEventListener('click',async()=>{await saveDraft();await act('question',{trial:state.trial,field:button.dataset.question});}));
  document.querySelector('#decision')?.addEventListener('submit',e=>{e.preventDefault();const data=gather();if(data.action==='order'&&data.q===null)return error('최종 발주량을 입력하세요.');act('submit',{trial:state.trial,...data});});
  if(state.phase==='task') {
    worksheet();
    document.querySelector('#decision').addEventListener('input',()=>{worksheet();clearTimeout(draftTimer);draftTimer=setTimeout(saveDraft,600);});
    document.querySelector('#use-fields')?.addEventListener('click',()=>{for(const p of state.advice.parameters){if(!['c','p','v'].includes(p.name)||p.value===null)continue;document.querySelector('#value-'+p.name).value=p.value;document.querySelector('#source-'+p.name).value=typeof p.source==='string'?p.source:'';}worksheet();saveDraft();});
    document.querySelector('#use-quantity')?.addEventListener('click',()=>{document.querySelector('input[name="action"][value="order"]').checked=true;document.querySelector('#quantity').value=Number(state.advice.q.toFixed(4));worksheet();saveDraft();});
  }
}
document.querySelector('#withdraw').addEventListener('click',()=>{if(confirm('참여를 중단하고 과제 응답을 삭제할까요? 다시 진행할 수 없습니다.'))act('withdraw');});
(async()=>{
  try {const token=new URLSearchParams(location.hash.slice(1)).get('join');if(token){state=await api('/api/join',{token});history.replaceState(null,'',location.pathname);}else{state=await api('/api/state',undefined,'GET');}render();}
  catch(e){root.innerHTML=`<section class="card narrow"><h1>개인 초대 링크가 필요합니다.</h1><p>진행자가 전달한 링크로 접속해 주세요. 같은 링크로 다시 접속하면 이어서 진행할 수 있습니다.</p></section>`;error(e.message);}
})();

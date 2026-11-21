const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  system: null,
  datasets: [],
  adapters: [],
  selectedDataset: null,
  selectedAdapter: null,
  activeJob: null,
  logOffset: 0,
  chat: [],
  streaming: false,
};

// ---------------------------------------------------------------- 基础工具
async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let message = `请求失败（${res.status}）`;
    try {
      const data = await res.json();
      if (data && data.detail) message = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
    } catch (_) { /* 忽略非 JSON 错误体 */ }
    throw new Error(message);
  }
  const type = res.headers.get('content-type') || '';
  return type.includes('application/json') ? res.json() : res.text();
}

let toastTimer = null;
function toast(message, bad = false) {
  const el = $('#toast');
  el.textContent = message;
  el.classList.toggle('bad', bad);
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 3200);
}

function escapeHtml(text) {
  return String(text ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function fmtTime(value) {
  return value ? String(value).slice(5, 16) : '';
}

function fmtNumber(value) {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'number') {
    if (Math.abs(value) >= 1000) return value.toLocaleString('zh-CN', { maximumFractionDigits: 0 });
    return String(Math.round(value * 100) / 100);
  }
  return String(value);
}

// ---------------------------------------------------------------- 标签页
$$('.nav').forEach((btn) => {
  btn.addEventListener('click', () => {
    $$('.nav').forEach((item) => item.classList.remove('active'));
    $$('.panel').forEach((panel) => panel.classList.remove('show'));
    btn.classList.add('active');
    $(`#panel-${btn.dataset.tab}`).classList.add('show');
  });
});

// ---------------------------------------------------------------- 系统状态
async function loadSystem() {
  const data = await api('/api/system');
  state.system = data;

  const device = data.device || {};
  const deviceText = device.cuda ? `GPU · ${device.gpu}` : `CPU · ${device.threads} 线程`;
  $('#devicePill').textContent = deviceText;
  $('#devicePill').className = 'pill ok';

  const defaultModel = data.default_model;
  const modelInfo = (data.models || []).find((item) => item.id === defaultModel) || {};
  const pill = $('#modelPill');
  if (modelInfo.downloaded) {
    pill.textContent = `模型就绪 · ${Math.round((modelInfo.bytes || 0) / 1024 ** 2)} MB`;
    pill.className = 'pill ok';
  } else if (data.download && data.download.status === 'downloading') {
    pill.textContent = `下载中 ${data.download.message || ''}`;
    pill.className = 'pill warn';
  } else {
    pill.textContent = '基座模型未下载';
    pill.className = 'pill warn';
  }

  $('#downloadBtn').textContent = modelInfo.downloaded ? '重新校验模型' : '下载基座模型';

  const busy = data.busy || {};
  $('#taskHint').textContent = busy.busy ? `任务运行中：${busy.kind}` : '空闲';
  syncBusy(busy.busy);
  // 页面刷新或换设备打开时，自动接上正在跑的训练任务
  if (busy.busy && busy.kind === 'train' && busy.id && state.activeJob !== busy.id) {
    attachJob(busy.id);
  }

  fillModelSelects(data.models || [], defaultModel);
  return data;
}

function fillModelSelects(models, defaultModel) {
  const trainSelect = $('#trainModel');
  if (trainSelect.options.length !== models.length) {
    trainSelect.innerHTML = models
      .map((item) => `<option value="${item.id}">${item.id}${item.downloaded ? '' : '（未下载）'}</option>`)
      .join('');
    trainSelect.value = defaultModel;
  }
}

function syncBusy(busy) {
  $('#startTrainBtn').disabled = busy;
  $('#sendBtn').disabled = busy || state.streaming;
}

// 接管一个已在运行的任务：清空日志面板，从头拉取并持续轮询
function attachJob(jobId) {
  state.activeJob = jobId;
  state.logOffset = 0;
  $('#trainLog').textContent = '';
  $('#progressBar').style.width = '0%';
  $('#progressText').textContent = '0%';
  $('#trainState').textContent = '训练中';
  $('#trainState').className = 'pill warn';
  pollJob(jobId);
}

// ---------------------------------------------------------------- 数据集
async function refreshDatasets() {
  state.datasets = await api('/api/datasets');
  renderDatasets();
  renderDatasetOptions();
}

function renderDatasets() {
  const list = $('#datasetList');
  if (!state.datasets.length) {
    list.innerHTML = '<div class="empty">还没有数据集<br/>点击右上角导入文件，或载入示例数据</div>';
    return;
  }
  list.innerHTML = state.datasets
    .map((item) => `
      <div class="item ${state.selectedDataset === item.id ? 'active' : ''}" data-id="${item.id}">
        <h3>${escapeHtml(item.name)}<span class="tag">${item.num_samples} 条</span></h3>
        <div class="meta">
          <span>${fmtTime(item.created_at)}</span>
          ${item.stats && item.stats.skipped ? `<span>跳过 ${item.stats.skipped} 条</span>` : ''}
        </div>
        <div class="row-actions">
          <button class="btn sm" data-action="preview" data-id="${item.id}">查看样本</button>
          <button class="btn sm danger" data-action="delete" data-id="${item.id}">删除</button>
        </div>
      </div>`)
    .join('');
}

function renderDatasetOptions() {
  const select = $('#trainDataset');
  const current = select.value;
  select.innerHTML = state.datasets
    .map((item) => `<option value="${item.id}">${escapeHtml(item.name)}（${item.num_samples} 条）</option>`)
    .join('');
  if (current && state.datasets.some((item) => item.id === current)) select.value = current;
}

async function previewDataset(id) {
  state.selectedDataset = id;
  renderDatasets();
  const data = await api(`/api/datasets/${id}/preview?limit=20`);
  const detail = $('#datasetDetail');
  if (!data.samples.length) {
    detail.innerHTML = '<div class="empty">该数据集为空</div>';
    return;
  }
  detail.innerHTML = `
    <h3 style="font-size:13px;margin-bottom:12px;color:var(--muted)">共 ${data.total} 条 · 展示前 ${data.samples.length} 条</h3>
    ${data.samples.map((sample) => `
      <div class="compare">
        ${sample.messages.map((msg) => `
          <div class="col"><b>${msg.role}</b>${escapeHtml(msg.content)}</div>`).join('')}
      </div>`).join('')}`;
}

async function importFile(file) {
  const form = new FormData();
  form.append('file', file);
  toast('正在解析文件…');
  const result = await api('/api/datasets/import', { method: 'POST', body: form });
  toast(`导入成功：${result.num_samples} 条${result.skipped ? `，跳过 ${result.skipped} 条` : ''}`);
  await refreshDatasets();
  await previewDataset(result.id);
}

$('#fileInput').addEventListener('change', async (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  try {
    await importFile(file);
  } catch (err) {
    toast(err.message, true);
  } finally {
    event.target.value = '';
  }
});

$('#importDemoBtn').addEventListener('click', async () => {
  try {
    const result = await api('/api/datasets/demo', { method: 'POST' });
    toast(`示例数据已载入：${result.num_samples} 条`);
    await refreshDatasets();
    await previewDataset(result.id);
  } catch (err) {
    toast(err.message, true);
  }
});

$('#datasetList').addEventListener('click', async (event) => {
  const btn = event.target.closest('button');
  const item = event.target.closest('.item');
  const id = (btn && btn.dataset.id) || (item && item.dataset.id);
  if (!id) return;
  const action = btn ? btn.dataset.action : 'preview';
  try {
    if (action === 'delete') {
      if (!confirm('确定删除该数据集？')) return;
      await api(`/api/datasets/${id}`, { method: 'DELETE' });
      if (state.selectedDataset === id) {
        state.selectedDataset = null;
        $('#datasetDetail').innerHTML = '<div class="empty">选择左侧数据集查看样本</div>';
      }
      await refreshDatasets();
      toast('已删除');
      return;
    }
    await previewDataset(id);
  } catch (err) {
    toast(err.message, true);
  }
});

// ---------------------------------------------------------------- 训练
$('#trainForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  const datasetId = $('#trainDataset').value;
  if (!datasetId) {
    toast('请先选择数据集', true);
    return;
  }
  const payload = {
    dataset_id: datasetId,
    base_model: $('#trainModel').value,
    epochs: Number($('#epochs').value),
    max_steps: Number($('#maxSteps').value),
    batch_size: Number($('#batchSize').value),
    grad_accum: Number($('#gradAccum').value),
    lr: Number($('#lr').value),
    lora_r: Number($('#loraR').value),
    lora_alpha: Number($('#loraAlpha').value),
    max_seq_len: Number($('#maxSeqLen').value),
  };
  try {
    $('#trainLog').textContent = '正在创建任务…';
    state.logOffset = 0;
    const result = await api('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    state.activeJob = result.job_id;
    toast('训练任务已启动');
    pollJob(result.job_id);
    loadSystem().catch(() => {});
  } catch (err) {
    toast(err.message, true);
    $('#trainLog').textContent = '启动失败：' + err.message;
  }
});

async function pollJob(jobId) {
  let finished = false;
  while (!finished) {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      const log = await api(`/api/jobs/${jobId}/log?after=${state.logOffset}`);
      if (log.lines.length) {
        appendLog(log.lines);
        state.logOffset = log.next;
      }
      $('#progressBar').style.width = `${Math.min(100, job.progress || 0)}%`;
      $('#progressText').textContent = `${Math.round(job.progress || 0)}%`;
      $('#trainState').textContent = job.status === 'running' ? '训练中' : job.status === 'success' ? '训练完成' : '训练失败';
      $('#trainState').className = `pill ${job.status === 'success' ? 'ok' : job.status === 'failed' ? 'bad' : 'warn'}`;

      if (job.status !== 'running') {
        finished = true;
        appendLog([{ type: 'stage', text: `任务结束：${job.status} ${job.message || ''}` }]);
        if (job.status === 'success') {
          $('#progressBar').style.width = '100%';
          $('#progressText').textContent = '100%';
          toast('训练完成，适配器已生成');
          await refreshAdapters();
        } else {
          toast('训练失败：' + (job.message || ''), true);
        }
        loadSystem().catch(() => {});
      }
    } catch (err) {
      finished = true;
      toast(err.message, true);
    }
    if (!finished) await new Promise((resolve) => setTimeout(resolve, 1200));
  }
}

function appendLog(lines) {
  const box = $('#trainLog');
  if (box.textContent.startsWith('等待任务') || box.textContent.startsWith('正在创建')) box.textContent = '';
  const text = lines
    .map((line) => {
      if (line.type === 'step') {
        const loss = line.logs && line.logs.loss !== undefined ? ` loss=${line.logs.loss}` : '';
        return `[${line.ts}] step ${line.step}/${line.max_steps}${loss}`;
      }
      if (line.type === 'error') return `[${line.ts}] ERROR ${line.text}`;
      return `[${line.ts}] ${line.text || ''}`;
    })
    .join('\n');
  box.textContent += (box.textContent ? '\n' : '') + text;
  box.scrollTop = box.scrollHeight;
}

// ---------------------------------------------------------------- 适配器 & 评测
async function refreshAdapters() {
  state.adapters = await api('/api/adapters');
  renderAdapters();
  renderChatAdapters();
}

function renderAdapters() {
  const list = $('#adapterList');
  if (!state.adapters.length) {
    list.innerHTML = '<div class="empty">还没有适配器<br/>先完成一次训练</div>';
    return;
  }
  list.innerHTML = state.adapters
    .map((item) => {
      const meta = item.meta || {};
      return `
      <div class="item ${state.selectedAdapter === item.id ? 'active' : ''}" data-id="${item.id}">
        <h3>${escapeHtml(item.name)}<span class="tag">r=${meta.lora_r || '-'}</span></h3>
        <div class="meta">
          <span>${escapeHtml(item.base_model || '')}</span>
          <span>${meta.train_samples || 0} 条</span>
          <span>loss ${fmtNumber(meta.train_loss)}</span>
        </div>
        <div class="meta">
          <span>${fmtTime(item.created_at)}</span>
          <span>${meta.duration_s ? meta.duration_s + 's' : ''}</span>
        </div>
        <div class="row-actions">
          <button class="btn sm" data-action="eval" data-id="${item.id}">评测</button>
          <button class="btn sm danger" data-action="delete" data-id="${item.id}">删除</button>
        </div>
      </div>`;
    })
    .join('');
}

$('#refreshAdapters').addEventListener('click', () => refreshAdapters().catch((err) => toast(err.message, true)));

$('#adapterList').addEventListener('click', async (event) => {
  const btn = event.target.closest('button');
  const item = event.target.closest('.item');
  const id = (btn && btn.dataset.id) || (item && item.dataset.id);
  if (!id) return;
  const action = btn ? btn.dataset.action : 'info';
  try {
    if (action === 'delete') {
      if (!confirm('确定删除该适配器？对应的训练产物会被移除。')) return;
      await api(`/api/adapters/${id}`, { method: 'DELETE' });
      await refreshAdapters();
      toast('已删除适配器');
      return;
    }
    if (action === 'eval') {
      await runEval(id);
      return;
    }
    state.selectedAdapter = id;
    renderAdapters();
  } catch (err) {
    toast(err.message, true);
  }
});

async function runEval(adapterId) {
  state.selectedAdapter = adapterId;
  renderAdapters();
  $('#evalDetail').innerHTML = '<div class="empty">正在评测：会加载基座模型并对比困惑度，请稍候…</div>';
  const { eval_id: evalId } = await api('/api/eval', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ adapter_id: adapterId }),
  });
  await loadSystem().catch(() => {});

  let finished = false;
  while (!finished) {
    const record = await api(`/api/evals/${evalId}`);
    if (record.status !== 'running') {
      finished = true;
      if (record.status === 'success') {
        renderEval(record.report);
        toast('评测完成');
      } else {
        $('#evalDetail').innerHTML = `<div class="empty">评测失败：${escapeHtml(record.report.error || '')}</div>`;
        toast('评测失败', true);
      }
    } else {
      await new Promise((resolve) => setTimeout(resolve, 1500));
    }
  }
  loadSystem().catch(() => {});
}

function renderEval(report) {
  const ppl = report.perplexity || {};
  const improvement = report.improvement;
  const improvementClass = improvement === null || improvement === undefined ? '' : improvement > 0 ? 'good' : '';
  const comparisons = (report.comparisons || [])
    .map((item) => `
      <div class="compare">
        <header>${escapeHtml(item.prompt)}</header>
        <div class="cols">
          <div class="col"><b>基座模型</b>${escapeHtml(item.base)}</div>
          <div class="col"><b>微调后</b>${escapeHtml(item.tuned)}</div>
        </div>
      </div>`)
    .join('');

  $('#evalDetail').innerHTML = `
    <div class="eval-grid">
      <div class="metric"><span>基座困惑度</span><strong>${fmtNumber(ppl.base)}</strong></div>
      <div class="metric"><span>微调后困惑度</span><strong>${fmtNumber(ppl.tuned)}</strong></div>
      <div class="metric"><span>下降幅度</span><strong class="${improvementClass}">${improvement === null || improvement === undefined ? '—' : improvement + '%'}</strong></div>
    </div>
    <p class="hint" style="margin-bottom:14px">验证集 ${report.eval_samples} 条 · 耗时 ${report.duration_s}s · 困惑度越低越好</p>
    ${comparisons || '<div class="empty">没有可用的对照样本</div>'}`;
}

// ---------------------------------------------------------------- 对话
function renderChatAdapters() {
  const select = $('#chatAdapter');
  const current = select.value;
  select.innerHTML = '<option value="">基座模型</option>' +
    state.adapters.map((item) => `<option value="${item.id}">${escapeHtml(item.name)}</option>`).join('');
  if (current && (current === '' || state.adapters.some((item) => item.id === current))) {
    select.value = current;
  } else if (state.adapters.length) {
    select.value = state.adapters[0].id;
  }

  const forgeSelect = $('#forgeAdapter');
  const forgeCurrent = forgeSelect.value;
  forgeSelect.innerHTML = '<option value="">基座模型</option>' +
    state.adapters.map((item) => `<option value="${item.id}">${escapeHtml(item.name)}</option>`).join('');
  if (forgeCurrent && state.adapters.some((item) => item.id === forgeCurrent)) {
    forgeSelect.value = forgeCurrent;
  } else if (state.adapters.length) {
    forgeSelect.value = state.adapters[0].id;
  }
}

function addMessage(role, content) {
  const box = $('#messages');
  const empty = box.querySelector('.empty');
  if (empty) empty.remove();
  const el = document.createElement('div');
  el.className = `msg ${role}`;
  el.innerHTML = `<div class="avatar">${role === 'user' ? '我' : 'AI'}</div><div class="bubble"></div>`;
  el.querySelector('.bubble').textContent = content;
  box.appendChild(el);
  box.scrollTop = box.scrollHeight;
  return el.querySelector('.bubble');
}

$('#clearChat').addEventListener('click', () => {
  state.chat = [];
  $('#messages').innerHTML = '<div class="empty">输入问题开始对话</div>';
});

// 数据飞轮：把这条回答（可先修正）存成训练样本
function addFeedbackButton(bubble, question, answer) {
  if (!answer) return;
  const row = document.createElement('div');
  row.className = 'msg-actions';
  const button = document.createElement('button');
  button.className = 'btn sm ghost';
  button.textContent = '存为样本';
  button.addEventListener('click', async () => {
    const edited = window.prompt('确认或修正后保存为训练样本：', answer);
    if (!edited || !edited.trim()) return;
    try {
      await api('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, answer: edited.trim() }),
      });
      button.textContent = '已存入';
      button.disabled = true;
      toast('已加入训练样本，下次训练生效');
    } catch (err) {
      toast(err.message, true);
    }
  });
  row.appendChild(button);
  bubble.parentElement.appendChild(row);
}

$('#temperature').addEventListener('input', (event) => {
  $('#tempValue').textContent = event.target.value;
});

$('#chatInput').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    $('#chatForm').requestSubmit();
  }
});

$('#chatForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (state.streaming) return;
  const input = $('#chatInput');
  const content = input.value.trim();
  if (!content) return;

  input.value = '';
  state.chat.push({ role: 'user', content });
  addMessage('user', content);
  const bubble = addMessage('assistant', '');
  bubble.classList.add('typing');
  const msgEl = bubble.parentElement;
  const body = document.createElement('div');
  body.className = 'msg-body';
  msgEl.insertBefore(body, bubble);
  const think = document.createElement('details');
  think.className = 'think';
  think.open = true;
  think.innerHTML = '<summary>思考过程</summary><div class="think-stream"></div>';
  body.appendChild(think);
  body.appendChild(bubble);
  const stream = think.querySelector('.think-stream');
  const addThink = (text, hits) => {
    const line = document.createElement('div');
    line.className = 'think-line';
    line.textContent = '· ' + text;
    stream.appendChild(line);
    (hits || []).forEach((hit) => {
      const sub = document.createElement('div');
      sub.className = 'think-hit';
      const score = hit.score != null ? `（相关度 ${Number(hit.score).toFixed(2)}）` : '';
      sub.textContent = `↳ ${hit.question}${score}`;
      stream.appendChild(sub);
    });
    $('#messages').scrollTop = $('#messages').scrollHeight;
  };
  const appendThought = (text) => {
    let node = stream.querySelector('.think-raw');
    if (!node) {
      node = document.createElement('div');
      node.className = 'think-raw md-body';
      node.dataset.raw = '';
      stream.appendChild(node);
    }
    node.dataset.raw += text;
    node.innerHTML = window.renderMarkdown ? window.renderMarkdown(node.dataset.raw) : node.dataset.raw;
    $('#messages').scrollTop = $('#messages').scrollHeight;
  };

  const payload = {
    messages: state.chat.slice(-12),
    adapter_id: $('#chatAdapter').value || null,
    use_adapter: $('#chatUseAdapter').checked,
    show_thought: $('#chatThought').checked,
    auto_approve: $('#chatAutoApprove').checked,
    verbosity: $('#chatVerbosity').value,
    max_new_tokens: 4096,
    temperature: Number($('#temperature').value),
  };

  state.streaming = true;
  $('#sendBtn').disabled = true;
  let answer = '';

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      let message = `请求失败（${res.status}）`;
      try {
        const data = await res.json();
        if (data.detail) message = data.detail;
      } catch (_) { /* ignore */ }
      throw new Error(message);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let done = false;
    while (!done) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      let index = buffer.indexOf('\n\n');
      while (index >= 0) {
        const raw = buffer.slice(0, index);
        buffer = buffer.slice(index + 2);
        index = buffer.indexOf('\n\n');
        const line = raw.split('\n').find((item) => item.startsWith('data:'));
        if (!line) continue;
        const event = JSON.parse(line.slice(5));
        if (event.type === 'delta') {
          if (!think.dataset.collapsed) {
            think.dataset.collapsed = '1';
            think.open = false;
          }
          answer += event.text;
          bubble.innerHTML = window.renderMarkdown ? window.renderMarkdown(answer) : answer;
          $('#messages').scrollTop = $('#messages').scrollHeight;
        } else if (event.type === 'status') {
          addThink(event.text);
        } else if (event.type === 'thinking') {
          addThink(event.text, event.hits);
        } else if (event.type === 'thought') {
          appendThought(event.text);
        } else if (event.type === 'ready') {
          bubble.textContent = '';
        } else if (event.type === 'done') {
          done = true;
        } else if (event.type === 'error') {
          throw new Error(event.message);
        }
      }
    }
    state.chat.push({ role: 'assistant', content: answer });
    bubble.innerHTML = window.renderMarkdown ? window.renderMarkdown(answer) : answer;
    addFeedbackButton(bubble, content, answer);
  } catch (err) {
    bubble.textContent = '出错了：' + err.message;
    toast(err.message, true);
  } finally {
    bubble.classList.remove('typing');
    state.streaming = false;
    $('#sendBtn').disabled = false;
  }
});

// ---------------------------------------------------------------- 文本工坊
const forgeState = { tasks: [], selected: null, output: '' };

async function loadForgeTasks() {
  try {
    forgeState.tasks = await api('/api/forge/tasks');
    renderForgeTasks();
    if (!forgeState.selected && forgeState.tasks.length) selectForgeTask(forgeState.tasks[0].id);
  } catch (err) {
    toast('加载任务类型失败：' + err.message, true);
  }
}

function renderForgeTasks() {
  const list = $('#forgeTasks');
  if (!forgeState.tasks.length) {
    list.innerHTML = '<div class="empty">没有可用的任务类型</div>';
    return;
  }
  list.innerHTML = forgeState.tasks
    .map((task) => `
      <div class="item ${forgeState.selected === task.id ? 'active' : ''}" data-id="${task.id}">
        <h3>${escapeHtml(task.name)}<span class="tag">${escapeHtml(task.output_format)}</span></h3>
        <div class="meta"><span>${escapeHtml(task.description)}</span></div>
      </div>`)
    .join('');
}

function selectForgeTask(id) {
  const task = forgeState.tasks.find((item) => item.id === id);
  if (!task) return;
  forgeState.selected = id;
  renderForgeTasks();
  $('#forgeTaskName').textContent = task.name;
  $('#forgeFormat').textContent = task.output_format;
  $('#forgeInput').placeholder = task.input_hint || '在这里输入内容';
  $('#forgeTemp').value = task.temperature;
  $('#forgeTempValue').textContent = task.temperature;
  $('#forgeMax').value = task.max_tokens;
  $('#forgeVars').innerHTML = (task.variables || [])
    .map((variable) => `
      <label>${escapeHtml(variable.label)}
        <input type="text" data-var="${escapeHtml(variable.name)}" value="${escapeHtml(variable.default)}" />
      </label>`)
    .join('');
}

$('#forgeTasks').addEventListener('click', (event) => {
  const item = event.target.closest('.item');
  if (item) selectForgeTask(item.dataset.id);
});

$('#forgeTemp').addEventListener('input', (event) => {
  $('#forgeTempValue').textContent = event.target.value;
});

$('#forgeClear').addEventListener('click', () => {
  $('#forgeInput').value = '';
  forgeState.output = '';
  $('#forgeOut').textContent = '选择任务类型并输入内容，点击运行。';
  $('#forgeMeta').textContent = '待运行';
});

$('#forgeCopy').addEventListener('click', async () => {
  if (!forgeState.output) {
    toast('还没有可复制的内容', true);
    return;
  }
  try {
    await navigator.clipboard.writeText(forgeState.output);
    toast('已复制到剪贴板');
  } catch (_) {
    toast('复制失败，请手动选择文本', true);
  }
});

$('#forgeDownload').addEventListener('click', () => {
  if (!forgeState.output) {
    toast('还没有可下载的内容', true);
    return;
  }
  const task = forgeState.tasks.find((item) => item.id === forgeState.selected);
  const ext = task && task.output_format === 'json' ? 'json' : 'txt';
  const blob = new Blob([forgeState.output], { type: 'text/plain;charset=utf-8' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `${task ? task.id : 'output'}-${Date.now()}.${ext}`;
  link.click();
  URL.revokeObjectURL(link.href);
});

$('#forgeRun').addEventListener('click', async () => {
  const taskId = forgeState.selected;
  if (!taskId) {
    toast('请先选择任务类型', true);
    return;
  }
  const input = $('#forgeInput').value.trim();
  if (!input) {
    toast('请输入内容', true);
    return;
  }
  const variables = {};
  $$('#forgeVars input[data-var]').forEach((element) => {
    if (element.value.trim()) variables[element.dataset.var] = element.value.trim();
  });

  const payload = {
    task_id: taskId,
    input,
    variables,
    adapter_id: $('#forgeAdapter').value || null,
    temperature: Number($('#forgeTemp').value),
    max_tokens: Number($('#forgeMax').value),
  };

  const output = $('#forgeOut');
  output.textContent = '';
  $('#forgeMeta').textContent = '生成中…';
  $('#forgeRun').disabled = true;

  try {
    const res = await fetch('/api/forge/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      let message = `请求失败（${res.status}）`;
      try {
        const data = await res.json();
        if (data.detail) message = data.detail;
      } catch (_) { /* ignore */ }
      throw new Error(message);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finished = false;
    while (!finished) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      let index = buffer.indexOf('\n\n');
      while (index >= 0) {
        const raw = buffer.slice(0, index);
        buffer = buffer.slice(index + 2);
        index = buffer.indexOf('\n\n');
        const line = raw.split('\n').find((item) => item.startsWith('data:'));
        if (!line) continue;
        const event = JSON.parse(line.slice(5));
        if (event.type === 'delta') {
          output.textContent += event.text;
          output.scrollTop = output.scrollHeight;
        } else if (event.type === 'reset') {
          output.textContent = '';
        } else if (event.type === 'status') {
          $('#forgeMeta').textContent = event.text;
        } else if (event.type === 'final') {
          output.textContent = event.text;
          forgeState.output = event.text;
          $('#forgeMeta').textContent = `${event.format} · ${event.elapsed}s${event.valid ? '' : ' · 格式待修正'}`;
          finished = true;
        } else if (event.type === 'error') {
          throw new Error(event.message);
        }
      }
    }
  } catch (err) {
    output.textContent = '出错了：' + err.message;
    toast(err.message, true);
  } finally {
    $('#forgeRun').disabled = false;
  }
});

// ---------------------------------------------------------------- 沙盒
const sbState = { path: '', pending: [] };

async function loadSandbox() {
  try {
    const info = await api('/api/sandbox/info');
    $('#sbWorkspace').textContent = info.workspace;
    await refreshSbFiles('');
    await refreshPending();
  } catch (err) {
    toast('沙盒初始化失败：' + err.message, true);
  }
}

async function refreshSbFiles(path) {
  sbState.path = path || '';
  const data = await api(`/api/sandbox/files?path=${encodeURIComponent(sbState.path)}`);
  const items = data.items || [];
  const up = sbState.path ? '<div class="item" data-up="1"><h3>↩ 返回上级</h3></div>' : '';
  $('#sbFiles').innerHTML = up + (items.map((item) => `
    <div class="item" data-path="${escapeHtml(item.path)}" data-dir="${item.dir}">
      <h3>${item.dir ? '📁' : '📄'} ${escapeHtml(item.name)}</h3>
      <div class="meta">${item.dir ? '<span>目录</span>' : `<span>${Math.max(1, Math.round(item.size / 1024))} KB</span>`}</div>
    </div>`).join('') || (up ? '' : '<div class="empty">工作区为空，直接写第一个文件吧</div>'));
}

$('#sbFiles').addEventListener('click', async (event) => {
  const item = event.target.closest('.item');
  if (!item) return;
  if (item.dataset.up) {
    const parent = sbState.path.split('/').slice(0, -1).join('/');
    return refreshSbFiles(parent).catch((err) => toast(err.message, true));
  }
  if (item.dataset.dir === 'true') {
    return refreshSbFiles(item.dataset.path).catch((err) => toast(err.message, true));
  }
  try {
    const data = await api(`/api/sandbox/file?path=${encodeURIComponent(item.dataset.path)}`);
    $('#sbFilePath').textContent = data.path;
    $('#sbNewPath').value = data.path;
    $('#sbEditor').value = data.content;
  } catch (err) {
    toast(err.message, true);
  }
});

$('#sbRefresh').addEventListener('click', () => loadSandbox().catch((err) => toast(err.message, true)));

$('#sbSave').addEventListener('click', async () => {
  const path = ($('#sbNewPath').value || '').trim();
  if (!path) {
    toast('请填写文件路径', true);
    return;
  }
  try {
    await api('/api/sandbox/file', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, content: $('#sbEditor').value }),
    });
    $('#sbFilePath').textContent = path;
    toast('已保存 ' + path);
    refreshSbFiles(sbState.path).catch(() => {});
  } catch (err) {
    toast(err.message, true);
  }
});

function renderSbResult(result) {
  const parts = [];
  if (result.stdout) parts.push(result.stdout.trimEnd());
  if (result.stderr) parts.push('[stderr]\n' + result.stderr.trimEnd());
  if (!parts.length) parts.push(result.status === 'denied' ? '已拒绝：' + result.reason : '（无输出）');
  $('#sbOut').textContent = parts.join('\n\n');
  $('#sbMeta').textContent = result.status === 'executed'
    ? `${result.elapsed}s · exit ${result.code}`
    : (result.reason || result.status);
}

$('#sbRun').addEventListener('click', async () => {
  const command = $('#sbCommand').value.trim();
  if (!command) {
    toast('请输入命令', true);
    return;
  }
  $('#sbOut').textContent = '执行中…';
  try {
    const result = await api('/api/sandbox/exec', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command }),
    });
    if (result.status === 'needs_approval') {
      $('#sbOut').textContent = '该命令需要你批准后才能执行：' + result.request.reason;
      toast('已生成授权请求', true);
      await refreshPending();
      return;
    }
    renderSbResult(result);
    refreshSbFiles(sbState.path).catch(() => {});
  } catch (err) {
    $('#sbOut').textContent = '出错了：' + err.message;
  }
});

$('#sbCommand').addEventListener('keydown', (event) => {
  if (event.key === 'Enter') $('#sbRun').click();
});

async function refreshPending() {
  const items = await api('/api/sandbox/pending');
  $('#sbPendingCount').textContent = `${items.length} 个待批准`;
  $('#sbPendingList').innerHTML = items.map((item) => `
    <div class="pending-item">
      <div>请求：<code>${escapeHtml(item.command)}</code></div>
      <div class="meta">原因：${escapeHtml(item.reason)} · ${item.created_at}</div>
      <div class="row-actions">
        <button class="btn sm" data-approve="${item.id}">允许执行</button>
        <button class="btn sm danger" data-deny="${item.id}">拒绝</button>
      </div>
    </div>`).join('');
}

$('#sbPendingList').addEventListener('click', async (event) => {
  const approveBtn = event.target.closest('[data-approve]');
  const denyBtn = event.target.closest('[data-deny]');
  try {
    if (approveBtn) {
      const result = await api(`/api/sandbox/approve/${approveBtn.dataset.approve}`, { method: 'POST' });
      renderSbResult(result);
      toast('已执行');
    } else if (denyBtn) {
      await api(`/api/sandbox/deny/${denyBtn.dataset.deny}`, { method: 'POST' });
      toast('已拒绝');
    } else {
      return;
    }
    await refreshPending();
    refreshSbFiles(sbState.path).catch(() => {});
  } catch (err) {
    toast(err.message, true);
  }
});

// ---------------------------------------------------------------- 模型下载
$('#downloadBtn').addEventListener('click', async () => {
  try {
    const modelId = $('#trainModel').value || (state.system && state.system.default_model);
    await api('/api/models/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: modelId }),
    });
    toast('开始下载模型（走 hf-mirror 镜像）');
    const timer = setInterval(async () => {
      const data = await loadSystem().catch(() => null);
      if (!data) return;
      if (data.download.status === 'ready') {
        clearInterval(timer);
        toast('模型下载完成');
      } else if (data.download.status === 'error') {
        clearInterval(timer);
        toast('下载失败：' + data.download.message, true);
      }
    }, 3000);
  } catch (err) {
    toast(err.message, true);
  }
});

$('#unloadBtn').addEventListener('click', async () => {
  try {
    await api('/api/runtime/unload', { method: 'POST' });
    toast('已释放模型权重');
    loadSystem().catch(() => {});
  } catch (err) {
    toast(err.message, true);
  }
});

// ---------------------------------------------------------------- 启动
async function checkAccess() {
  try {
    const data = await api('/api/access');
    state.admin = data.admin !== false;
  } catch (_) {
    // 网关未启用时不会拒绝，按管理员处理
    state.admin = true;
  }
  if (!state.admin) applyGuestMode();
}

function applyGuestMode() {
  ['data', 'train', 'adapters'].forEach((tab) => {
    const btn = document.querySelector(`.nav[data-tab="${tab}"]`);
    if (btn) btn.hidden = true;
    const panel = document.getElementById(`panel-${tab}`);
    if (panel) panel.classList.remove('show');
  });
  const downloadBtn = $('#downloadBtn');
  const unloadBtn = $('#unloadBtn');
  if (downloadBtn) downloadBtn.hidden = true;
  if (unloadBtn) unloadBtn.hidden = true;
  const chatBtn = document.querySelector('.nav[data-tab="chat"]');
  if (chatBtn) chatBtn.click();
  // 隐藏后把剩下的入口重新编号为 01、02、03
  $$('.nav')
    .filter((btn) => !btn.hidden)
    .forEach((btn, index) => {
      const span = btn.querySelector('span');
      if (span) span.textContent = String(index + 1).padStart(2, '0');
    });
}

(async function boot() {
  try {
    await checkAccess();
    const system = await loadSystem();
    if (state.admin) {
      await refreshDatasets();
    }
    // 适配器列表对访客只读，对话页的模型下拉需要它
    await refreshAdapters();
    await loadForgeTasks();
    await loadSandbox();
    if (system.train_defaults) {
      const defaults = system.train_defaults;
      if (defaults.epochs) $('#epochs').value = defaults.epochs;
      if (defaults.batch_size) $('#batchSize').value = defaults.batch_size;
      if (defaults.grad_accum) $('#gradAccum').value = defaults.grad_accum;
      if (defaults.lr) $('#lr').value = defaults.lr;
      if (defaults.lora_r) $('#loraR').value = defaults.lora_r;
      if (defaults.lora_alpha) $('#loraAlpha').value = defaults.lora_alpha;
      if (defaults.max_seq_len) $('#maxSeqLen').value = defaults.max_seq_len;
    }
  } catch (err) {
    toast('初始化失败：' + err.message, true);
  }
})();

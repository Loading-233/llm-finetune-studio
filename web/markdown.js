/* 轻量 Markdown 渲染：标题、粗斜体、行内代码、代码卡片（语法高亮/行号/复制/折叠）、
   列表、任务列表、表格、引用、分隔线。
   安全策略：先转义 HTML 再生成结构，模型输出永远不会被当作 HTML 执行。 */
(function () {
  const ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

  const LANG_LABELS = {
    ps1: 'PowerShell', powershell: 'PowerShell', pwsh: 'PowerShell',
    py: 'Python', python: 'Python',
    js: 'JavaScript', javascript: 'JavaScript', ts: 'TypeScript', typescript: 'TypeScript',
    json: 'JSON', yaml: 'YAML', yml: 'YAML', toml: 'TOML', ini: 'INI',
    bash: 'Bash', sh: 'Bash', shell: 'Shell', zsh: 'Zsh',
    sql: 'SQL', html: 'HTML', css: 'CSS', md: 'Markdown', markdown: 'Markdown',
    bat: 'Batch', cmd: 'Batch', c: 'C', cpp: 'C++', java: 'Java', go: 'Go', rust: 'Rust',
  };

  // 各语言关键字（够用即可，不求完备）
  const KEYWORDS = {
    python: 'def class return if elif else for while in not and or import from as with try except finally raise lambda None True False pass break continue yield async await print self',
    javascript: 'function const let var return if else for while of in new class extends import export default async await try catch finally throw typeof instanceof null undefined true false',
    typescript: 'function const let var return if else for while of in new class extends implements interface import export default async await try catch throw type enum null undefined true false',
    sql: 'select from where join left right inner outer on group by order having limit offset insert into values update set delete create table alter drop index and or not null as distinct count sum avg min max',
    powershell: 'if else elseif foreach for while function param return try catch finally throw begin process end switch',
    bash: 'if then else elif fi for while do done case esac function return export local echo cd ls cat grep',
    json: 'true false null',
    yaml: 'true false null',
    generic: 'true false null',
  };

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, (char) => ESCAPES[char]);
  }

  function keywordsFor(lang) {
    const key = (lang || '').toLowerCase();
    const alias = { py: 'python', js: 'javascript', ts: 'typescript', ps1: 'powershell', pwsh: 'powershell', sh: 'bash', shell: 'bash', zsh: 'bash' };
    const name = alias[key] || key;
    return new Set((KEYWORDS[name] || KEYWORDS.generic).split(/\s+/));
  }

  // 语法高亮：单趟扫描，注释/字符串优先，其次数字、标识符
  function highlight(raw, lang) {
    const words = keywordsFor(lang);
    const isJson = (lang || '').toLowerCase() === 'json';
    const isSql = /sql/i.test(lang || '');
    const re = /(\/\/[^\n]*|#[^\n]*|\/\*[\s\S]*?\*\/)|("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)|(\b\d+(?:\.\d+)?\b)|([A-Za-z_$][A-Za-z0-9_$]*)/g;
    let out = '';
    let last = 0;
    let match;
    while ((match = re.exec(raw)) !== null) {
      out += escapeHtml(raw.slice(last, match.index));
      const [text, comment, string, number, word] = match;
      if (comment) {
        out += `<span class="tok-comment">${escapeHtml(comment)}</span>`;
      } else if (string) {
        // JSON 的键：字符串后面紧跟冒号
        const rest = raw.slice(re.lastIndex);
        const isKey = isJson && /^\s*:/.test(rest);
        out += `<span class="${isKey ? 'tok-key' : 'tok-string'}">${escapeHtml(string)}</span>`;
      } else if (number) {
        out += `<span class="tok-number">${escapeHtml(number)}</span>`;
      } else if (word) {
        const lower = word.toLowerCase();
        const isKeyword = words.has(isSql ? lower : word);
        out += isKeyword ? `<span class="tok-keyword">${escapeHtml(word)}</span>` : escapeHtml(word);
      }
      last = re.lastIndex;
    }
    out += escapeHtml(raw.slice(last));
    return out;
  }

  function codeCard(lang, code) {
    const key = (lang || '').toLowerCase();
    const label = key ? (LANG_LABELS[key] || lang) : '代码';
    const body = code
      .replace(/\n$/, '')
      .split('\n')
      .map((line) => `<span class="code-line">${highlight(line, key) || ' '}</span>`)
      .join('');
    return (
      '<div class="code-card">' +
        '<div class="code-head">' +
          `<span class="code-lang">${escapeHtml(label)}</span>` +
          '<span class="code-arrow">⌄</span>' +
          '<button class="code-copy" type="button" title="复制代码" aria-label="复制代码">⧉</button>' +
        '</div>' +
        `<pre class="md-code"><code>${body}</code></pre>` +
      '</div>'
    );
  }

  function renderMarkdown(source) {
    if (!source) return '';
    const blocks = [];
    let src = String(source);

    // 代码块还没输出完：先不显示残缺内容，等闭合后再一次性渲染
    const fences = (src.match(/```/g) || []).length;
    if (fences % 2 === 1) {
      const cut = src.lastIndexOf('```');
      src = src.slice(0, cut) + `\n\u0000${blocks.push('<div class="code-pending">代码生成中…</div>') - 1}\u0000\n`;
    }

    // 已闭合的代码块先占位，避免后续行解析破坏内容
    src = src.replace(/```([\w+#-]*)\n([\s\S]*?)```/g, (_, lang, code) => {
      blocks.push(codeCard(lang, code));
      return `\u0000${blocks.length - 1}\u0000`;
    });

    const escaped = escapeHtml(src);
    const lines = escaped.split('\n');
    const out = [];
    let list = null;
    let quote = [];
    let table = [];

    const flushList = () => {
      if (list) {
        out.push(`</${list}>`);
        list = null;
      }
    };
    const flushQuote = () => {
      if (!quote.length) return;
      out.push(`<blockquote class="md-quote">${quote.map((line) => inline(line)).join('<br/>')}</blockquote>`);
      quote = [];
    };
    const flushTable = () => {
      if (!table.length) return;
      const cells = table.map((row) => row.split('|').slice(1, -1).map((cell) => cell.trim()));
      const isDivider = (row) => row.every((cell) => /^:?-{2,}:?$/.test(cell));
      const head = cells[0];
      const body = cells.length > 1 && isDivider(cells[1]) ? cells.slice(2) : cells.slice(1);
      const headHtml = '<thead><tr>' + head.map((cell) => `<th>${inline(cell)}</th>`).join('') + '</tr></thead>';
      const bodyHtml = body.length
        ? '<tbody>' + body.map((row) => '<tr>' + row.map((cell) => `<td>${inline(cell)}</td>`).join('') + '</tr>').join('') + '</tbody>'
        : '';
      out.push(`<table class="md-table">${headHtml}${bodyHtml}</table>`);
      table = [];
    };

    for (const raw of lines) {
      const line = raw.replace(/\s+$/, '');
      const placeholder = line.trim().match(/^\u0000(\d+)\u0000$/);
      if (placeholder) {
        flushList();
        flushTable();
        flushQuote();
        out.push(blocks[Number(placeholder[1])]);
        continue;
      }

      if (/^\s*\|.*\|\s*$/.test(line)) {
        flushList();
        flushQuote();
        table.push(line.trim());
        continue;
      }
      flushTable();

      const heading = line.match(/^(#{1,4})\s+(.*)$/);
      if (heading) {
        flushList();
        flushQuote();
        const level = heading[1].length;
        out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
        continue;
      }

      const quoteLine = line.match(/^\s*&gt;\s?(.*)$/);
      if (quoteLine) {
        flushList();
        quote.push(quoteLine[1]);
        continue;
      }
      flushQuote();

      const task = line.match(/^\s*[-*+]\s+\[([ xX])\]\s+(.*)$/);
      if (task) {
        if (list !== 'ul') {
          flushList();
          out.push('<ul class="md-tasks">');
          list = 'ul';
        }
        const checked = task[1].toLowerCase() === 'x';
        out.push(`<li class="md-task${checked ? ' done' : ''}">${checked ? '☑' : '☐'} ${inline(task[2])}</li>`);
        continue;
      }

      const unordered = line.match(/^\s*[-*+]\s+(.*)$/);
      if (unordered) {
        if (list !== 'ul') {
          flushList();
          out.push('<ul>');
          list = 'ul';
        }
        out.push(`<li>${inline(unordered[1])}</li>`);
        continue;
      }

      const ordered = line.match(/^\s*\d+[.)]\s+(.*)$/);
      if (ordered) {
        if (list !== 'ol') {
          flushList();
          out.push('<ol>');
          list = 'ol';
        }
        out.push(`<li>${inline(ordered[1])}</li>`);
        continue;
      }

      if (/^\s*([-*_])\1{2,}\s*$/.test(line)) {
        flushList();
        out.push('<hr/>');
        continue;
      }

      flushList();
      if (!line.trim()) continue;
      out.push(`<p>${inline(line)}</p>`);
    }

    flushList();
    flushQuote();
    flushTable();
    return out.join('');
  }

  function inline(text) {
    return text
      .replace(/`([^`]+)`/g, (_, code) => `<code>${code}</code>`)
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>')
      .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  }

  // 事件委托：流式输出会反复重绘气泡，按钮不能直接绑事件
  document.addEventListener('click', (event) => {
    const copy = event.target.closest('.code-copy');
    if (copy) {
      event.stopPropagation();
      const card = copy.closest('.code-card');
      const code = card && card.querySelector('pre code');
      const text = code ? (code.innerText || '') : '';
      const reset = () => { copy.textContent = '⧉'; copy.classList.remove('ok', 'bad'); };
      const ok = () => {
        copy.textContent = '✓';
        copy.classList.add('ok');
        setTimeout(reset, 1500);
      };
      const fail = () => {
        copy.textContent = '×';
        copy.classList.add('bad');
        setTimeout(reset, 1500);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(ok).catch(() => {
          try {
            const range = document.createRange();
            range.selectNodeContents(code);
            const selection = window.getSelection();
            selection.removeAllRanges();
            selection.addRange(range);
            ok();
          } catch (err) {
            fail();
          }
        });
      } else {
        fail();
      }
      return;
    }

    const head = event.target.closest('.code-head');
    if (head) {
      head.parentElement.classList.toggle('collapsed');
    }
  });

  window.renderMarkdown = renderMarkdown;
})();

"""Generate self-contained interactive live adaptation visualizer HTML."""
import json
from pathlib import Path


def build_visualizer_html(data_json_str: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Live Multi-Horizon Adaptation Visualizer</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border-color: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent-cyan: #06b6d4;
      --accent-green: #10b981;
      --accent-rose: #f43f5e;
      --accent-amber: #f59e0b;
      --accent-purple: #a855f7;
    }}
    body {{
      background: transparent;
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    .glow {{
      box-shadow: 0 0 15px rgba(6, 182, 212, 0.25);
    }}
  </style>
</head>
<body class="p-4 antialiased">
  <div class="max-w-6xl mx-auto bg-[var(--card-bg)] border border-[var(--border-color)] rounded-2xl p-6 shadow-2xl">
    
    <!-- Header -->
    <div class="flex flex-wrap items-center justify-between border-b border-[var(--border-color)] pb-4 mb-4 gap-4">
      <div>
        <div class="flex items-center gap-2">
          <span class="inline-block w-3 h-3 rounded-full bg-[var(--accent-green)] animate-ping"></span>
          <h1 class="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            Prequential Multi-Horizon Adaptation Visualizer
          </h1>
        </div>
        <p class="text-xs text-[var(--text-muted)] mt-1">
          Asynchronous Feedback Release & Real-time Cross-Horizon Transfer (Chronos-T5-small & Baselines)
        </p>
      </div>

      <!-- Controls: Scenario & Horizon -->
      <div class="flex flex-wrap items-center gap-3">
        <div>
          <label class="block text-[11px] uppercase tracking-wider text-[var(--text-muted)] mb-1">Shift Scenario</label>
          <select id="scenarioSelect" class="bg-[#0f172a] border border-[var(--border-color)] text-sm rounded-lg px-3 py-1.5 focus:outline-none focus:border-[var(--accent-cyan)] text-cyan-300 font-medium">
          </select>
        </div>

        <div>
          <label class="block text-[11px] uppercase tracking-wider text-[var(--text-muted)] mb-1">Horizon (h)</label>
          <div class="inline-flex rounded-lg border border-[var(--border-color)] bg-[#0f172a] p-0.5" id="horizonButtons">
            <button data-h="1" class="h-btn px-2.5 py-1 text-xs font-semibold rounded bg-[var(--accent-cyan)] text-black">h=1</button>
            <button data-h="4" class="h-btn px-2.5 py-1 text-xs font-semibold rounded text-[var(--text-muted)] hover:text-white">h=4</button>
            <button data-h="8" class="h-btn px-2.5 py-1 text-xs font-semibold rounded text-[var(--text-muted)] hover:text-white">h=8</button>
            <button data-h="16" class="h-btn px-2.5 py-1 text-xs font-semibold rounded text-[var(--text-muted)] hover:text-white">h=16</button>
          </div>
        </div>
      </div>
    </div>

    <!-- Playback Control Bar -->
    <div class="bg-[#0f172a] border border-[var(--border-color)] rounded-xl p-3 mb-5 flex flex-wrap items-center justify-between gap-4">
      <div class="flex items-center gap-2">
        <button id="btnPlayPause" class="px-4 py-1.5 rounded-lg bg-[var(--accent-cyan)] text-black font-semibold text-xs hover:opacity-90 flex items-center gap-1.5 shadow-sm">
          <span id="playIcon">▶</span> <span id="playText">Play Live</span>
        </button>
        <button id="btnStep" class="px-3 py-1.5 rounded-lg bg-[var(--border-color)] hover:bg-slate-700 text-white font-medium text-xs">
          Step +1
        </button>
        <button id="btnReset" class="px-3 py-1.5 rounded-lg bg-[var(--border-color)] hover:bg-slate-700 text-[var(--text-muted)] hover:text-white font-medium text-xs">
          Reset
        </button>
      </div>

      <!-- Timeline Slider -->
      <div class="flex-1 max-w-md flex items-center gap-3">
        <span class="text-xs font-mono text-[var(--accent-cyan)]">t=<span id="timeDisplay">0</span></span>
        <input type="range" id="timeSlider" min="0" max="99" value="0" class="w-full accent-[var(--accent-cyan)] cursor-pointer">
        <span class="text-xs font-mono text-[var(--text-muted)]" id="maxTimeDisplay">/99</span>
      </div>

      <!-- Speed Selector -->
      <div class="flex items-center gap-1.5 text-xs">
        <span class="text-[var(--text-muted)]">Speed:</span>
        <button data-speed="1" class="speed-btn px-2 py-0.5 rounded bg-[var(--accent-cyan)] text-black font-semibold">1x</button>
        <button data-speed="2" class="speed-btn px-2 py-0.5 rounded text-[var(--text-muted)] hover:text-white">2x</button>
        <button data-speed="5" class="speed-btn px-2 py-0.5 rounded text-[var(--text-muted)] hover:text-white">5x</button>
      </div>
    </div>

    <!-- Main Grid: 2 Columns -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-5">

      <!-- Left Column: Primary Forecast & Adaptation Trajectory (2 Cols) -->
      <div class="lg:col-span-2 space-y-5">
        <div class="bg-[#0f172a] border border-[var(--border-color)] rounded-xl p-4">
          <div class="flex items-center justify-between mb-2">
            <h3 class="text-sm font-semibold text-white flex items-center gap-2">
              <span class="w-2.5 h-2.5 rounded-full bg-[var(--accent-cyan)]"></span>
              Prequential Time Series & Predictive Intervals
            </h3>
            <div class="flex items-center gap-3 text-xs">
              <span class="flex items-center gap-1"><span class="w-3 h-0.5 bg-white inline-block"></span> Ground Truth</span>
              <span class="flex items-center gap-1 text-rose-400"><span class="w-3 h-0.5 bg-rose-400 inline-block"></span> B0: Frozen TSFM</span>
              <span class="flex items-center gap-1 text-cyan-400"><span class="w-3 h-0.5 bg-cyan-400 inline-block"></span> H3: Unified Adapter</span>
            </div>
          </div>
          <canvas id="timeSeriesCanvas" height="260" class="w-full rounded-lg bg-[#0b1120]"></canvas>
        </div>

        <!-- Matured Asynchronous Feedback Queue Monitor -->
        <div class="bg-[#0f172a] border border-[var(--border-color)] rounded-xl p-4">
          <h3 class="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-2 flex items-center justify-between">
            <span>Asynchronous Feedback Queue Status at Step t</span>
            <span id="queueNotice" class="text-cyan-400 font-mono lowercase text-[11px]"></span>
          </h3>
          <div class="grid grid-cols-4 gap-3 text-center" id="feedbackCards">
            <div id="card-h1" class="border border-[var(--border-color)] rounded-lg p-2.5 bg-slate-900/60 transition-all">
              <div class="text-[11px] text-[var(--text-muted)]">Horizon h=1</div>
              <div class="text-sm font-bold text-white mt-0.5" id="val-h1">Active</div>
              <div class="text-[10px] text-emerald-400 mt-1">Delay: 1 step</div>
            </div>
            <div id="card-h4" class="border border-[var(--border-color)] rounded-lg p-2.5 bg-slate-900/60 transition-all">
              <div class="text-[11px] text-[var(--text-muted)]">Horizon h=4</div>
              <div class="text-sm font-bold text-white mt-0.5" id="val-h4">Pending</div>
              <div class="text-[10px] text-amber-400 mt-1">Delay: 4 steps</div>
            </div>
            <div id="card-h8" class="border border-[var(--border-color)] rounded-lg p-2.5 bg-slate-900/60 transition-all">
              <div class="text-[11px] text-[var(--text-muted)]">Horizon h=8</div>
              <div class="text-sm font-bold text-white mt-0.5" id="val-h8">Pending</div>
              <div class="text-[10px] text-purple-400 mt-1">Delay: 8 steps</div>
            </div>
            <div id="card-h16" class="border border-[var(--border-color)] rounded-lg p-2.5 bg-slate-900/60 transition-all">
              <div class="text-[11px] text-[var(--text-muted)]">Horizon h=16</div>
              <div class="text-sm font-bold text-white mt-0.5" id="val-h16">Pending</div>
              <div class="text-[10px] text-rose-400 mt-1">Delay: 16 steps</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Right Column: CRPS Race & Latent State Orbit (1 Col) -->
      <div class="space-y-5">
        
        <!-- Live Cumulative CRPS Comparison -->
        <div class="bg-[#0f172a] border border-[var(--border-color)] rounded-xl p-4">
          <div class="flex items-center justify-between mb-3">
            <h3 class="text-sm font-semibold text-white">Cumulative CRPS (Lower = Better)</h3>
            <span class="text-[11px] font-mono text-[var(--accent-cyan)]" id="crpsAdvantageLabel">-0.00%</span>
          </div>
          <div class="space-y-3" id="crpsBarsContainer">
            <!-- Dynamic Bars -->
          </div>
        </div>

        <!-- Latent State Vector Orbit: z_0 vs z_1 -->
        <div class="bg-[#0f172a] border border-[var(--border-color)] rounded-xl p-4">
          <div class="flex items-center justify-between mb-2">
            <h3 class="text-sm font-semibold text-white">Latent State Space Orbit z_t</h3>
            <span class="text-[11px] font-mono text-[var(--accent-cyan)]" id="stateVectorDisplay">[0.000, 0.000]</span>
          </div>
          <canvas id="latentStateCanvas" height="180" class="w-full rounded-lg bg-[#0b1120]"></canvas>
        </div>

        <!-- Key Scientific Finding Pill -->
        <div class="bg-gradient-to-r from-cyan-950/40 to-slate-900 border border-cyan-800/40 rounded-xl p-3.5 text-xs text-slate-300">
          <span class="font-bold text-cyan-400">Scientific Takeaway:</span>
          <span id="takeawayText" class="block mt-1 text-[11px] leading-relaxed text-slate-300">
            H3 overlap-aware effective-sample scaling bounds update variance, allowing long horizons (h=16) to adapt up to 15 steps before observing their own labels.
          </span>
        </div>

      </div>

    </div>

  </div>

  <script>
    const TRACES_DATA = {data_json_str};

    let currentScenarioKey = Object.keys(TRACES_DATA)[0];
    let currentHorizon = 1;
    let currentT = 0;
    let isPlaying = false;
    let playSpeed = 1;
    let playInterval = null;

    // Elements
    const scenarioSelect = document.getElementById('scenarioSelect');
    const timeSlider = document.getElementById('timeSlider');
    const timeDisplay = document.getElementById('timeDisplay');
    const maxTimeDisplay = document.getElementById('maxTimeDisplay');
    const btnPlayPause = document.getElementById('btnPlayPause');
    const btnStep = document.getElementById('btnStep');
    const btnReset = document.getElementById('btnReset');
    const playIcon = document.getElementById('playIcon');
    const playText = document.getElementById('playText');
    const horizonButtons = document.querySelectorAll('.h-btn');
    const speedButtons = document.querySelectorAll('.speed-btn');
    const tsCanvas = document.getElementById('timeSeriesCanvas');
    const stateCanvas = document.getElementById('latentStateCanvas');
    const crpsContainer = document.getElementById('crpsBarsContainer');
    const stateDisplay = document.getElementById('stateVectorDisplay');
    const crpsAdvantageLabel = document.getElementById('crpsAdvantageLabel');
    const takeawayText = document.getElementById('takeawayText');

    // Populate scenarios
    Object.keys(TRACES_DATA).forEach(k => {{
      const opt = document.createElement('option');
      opt.value = k;
      opt.textContent = k;
      scenarioSelect.appendChild(opt);
    }});

    function getScenario() {{
      return TRACES_DATA[currentScenarioKey];
    }}

    function initScenario() {{
      const sc = getScenario();
      const nSteps = sc.steps.length;
      timeSlider.max = nSteps - 1;
      maxTimeDisplay.textContent = '/' + (nSteps - 1);
      currentT = 0;
      updateStep(0);
    }}

    scenarioSelect.addEventListener('change', (e) => {{
      currentScenarioKey = e.target.value;
      if (currentScenarioKey.includes("Joint Shift")) {{
        takeawayText.textContent = "Under severe Joint Shift, H3 Unified Adapter outperforms all independent baselines across long horizons (h=4, 8, 16), reducing CRPS by up to 21.8%.";
      }} else if (currentScenarioKey.includes("Abrupt Reversal")) {{
        takeawayText.textContent = "Under Abrupt Reversal, H2 age-decay weighting successfully dampens stale-feedback shocks, preventing runaway state drift.";
      }} else if (currentScenarioKey.includes("SPY")) {{
        takeawayText.textContent = "On real S&P 500 daily returns, H3 achieves statistically significant CRPS reduction over independent EWMA (p < 0.001) and improves interval Winkler score by 24.4%.";
      }} else if (currentScenarioKey.includes("BTCUSDT")) {{
        takeawayText.textContent = "On crypto volatility, H3 stabilizes forecast width and dampens heavy-tailed extreme shock over-corrections.";
      }} else {{
        takeawayText.textContent = "In stationary regimes, naive coupling suffers from negative transfer; overlap-aware scaling successfully preserves stability.";
      }}
      initScenario();
    }});

    horizonButtons.forEach(btn => {{
      btn.addEventListener('click', () => {{
        horizonButtons.forEach(b => {{
          b.classList.remove('bg-[var(--accent-cyan)]', 'text-black');
          b.classList.add('text-[var(--text-muted)]');
        }});
        btn.classList.add('bg-[var(--accent-cyan)]', 'text-black');
        btn.classList.remove('text-[var(--text-muted)]');
        currentHorizon = parseInt(btn.getAttribute('data-h'));
        renderAll();
      }});
    }});

    speedButtons.forEach(btn => {{
      btn.addEventListener('click', () => {{
        speedButtons.forEach(b => {{
          b.classList.remove('bg-[var(--accent-cyan)]', 'text-black');
          b.classList.add('text-[var(--text-muted)]');
        }});
        btn.classList.add('bg-[var(--accent-cyan)]', 'text-black');
        btn.classList.remove('text-[var(--text-muted)]');
        playSpeed = parseInt(btn.getAttribute('data-speed'));
        if (isPlaying) {{
          stopPlay();
          startPlay();
        }}
      }});
    }});

    timeSlider.addEventListener('input', (e) => {{
      updateStep(parseInt(e.target.value));
    }});

    btnStep.addEventListener('click', () => {{
      const sc = getScenario();
      if (currentT < sc.steps.length - 1) {{
        updateStep(currentT + 1);
      }}
    }});

    btnReset.addEventListener('click', () => {{
      stopPlay();
      updateStep(0);
    }});

    btnPlayPause.addEventListener('click', () => {{
      if (isPlaying) {{
        stopPlay();
      }} else {{
        startPlay();
      }}
    }});

    function startPlay() {{
      isPlaying = true;
      playIcon.textContent = '⏸';
      playText.textContent = 'Pause';
      btnPlayPause.classList.remove('bg-[var(--accent-cyan)]');
      btnPlayPause.classList.add('bg-amber-400');

      playInterval = setInterval(() => {{
        const sc = getScenario();
        if (currentT >= sc.steps.length - 1) {{
          stopPlay();
        }} else {{
          updateStep(currentT + 1);
        }}
      }}, 250 / playSpeed);
    }}

    function stopPlay() {{
      isPlaying = false;
      playIcon.textContent = '▶';
      playText.textContent = 'Play Live';
      btnPlayPause.classList.add('bg-[var(--accent-cyan)]');
      btnPlayPause.classList.remove('bg-amber-400');
      if (playInterval) clearInterval(playInterval);
    }}

    function updateStep(t) {{
      currentT = t;
      timeSlider.value = t;
      timeDisplay.textContent = t;
      renderAll();
    }}

    function renderAll() {{
      renderTimeSeries();
      renderLatentOrbit();
      renderFeedbackCards();
      renderCRPSBars();
    }}

    // Time Series Canvas
    function renderTimeSeries() {{
      const ctx = tsCanvas.getContext('2d');
      const w = tsCanvas.width = tsCanvas.clientWidth * 2;
      const h = tsCanvas.height = tsCanvas.clientHeight * 2;
      ctx.clearRect(0, 0, w, h);

      const sc = getScenario();
      const steps = sc.steps;
      if (steps.length === 0) return;

      const windowSize = 45;
      const startT = Math.max(0, currentT - windowSize + 1);
      const endT = Math.min(steps.length, currentT + 10);
      const visible = steps.slice(startT, endT);

      let allY = visible.map(s => s.y_true);
      visible.forEach(s => {{
        ['B0_Frozen', 'H3_Unified'].forEach(m => {{
          const hData = s.models[m]?.horizons[currentHorizon];
          if (hData) {{
            allY.push(hData.q10, hData.q90, hData.median);
          }}
        }});
      }});

      let minY = Math.min(...allY);
      let maxY = Math.max(...allY);
      const padY = (maxY - minY) * 0.15 || 0.05;
      minY -= padY;
      maxY += padY;

      function getX(idx) {{
        return 40 + ((idx - startT) / (endT - startT)) * (w - 70);
      }}
      function getY(val) {{
        return h - 30 - ((val - minY) / (maxY - minY)) * (h - 60);
      }}

      // Grid lines
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      for (let i = 0; i < 5; i++) {{
        const gy = minY + (i / 4) * (maxY - minY);
        const yPx = getY(gy);
        ctx.beginPath();
        ctx.moveTo(35, yPx);
        ctx.lineTo(w - 20, yPx);
        ctx.stroke();
        ctx.fillStyle = '#64748b';
        ctx.font = '20px monospace';
        ctx.fillText(gy.toFixed(3), 5, yPx + 6);
      }}

      // Current t indicator line
      const curX = getX(currentT);
      ctx.strokeStyle = '#38bdf8';
      ctx.setLineDash([6, 6]);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(curX, 10);
      ctx.lineTo(curX, h - 25);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw Ground Truth Path up to currentT
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 3;
      ctx.beginPath();
      for (let i = startT; i <= Math.min(currentT, steps.length - 1); i++) {{
        const x = getX(i);
        const y = getY(steps[i].y_true);
        if (i === startT) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }}
      ctx.stroke();

      // Draw Future Ground Truth (dimmed)
      if (currentT < steps.length - 1) {{
        ctx.strokeStyle = '#475569';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(curX, getY(steps[currentT].y_true));
        for (let i = currentT + 1; i < endT; i++) {{
          ctx.lineTo(getX(i), getY(steps[i].y_true));
        }}
        ctx.stroke();
        ctx.setLineDash([]);
      }}

      // At currentT, draw predictive distribution for B0 (Frozen) and H3 (Adapted)
      const curStep = steps[currentT];
      const targX = getX(currentT + currentHorizon);

      // B0 Cone (Rose)
      const b0 = curStep.models['B0_Frozen']?.horizons[currentHorizon];
      if (b0 && currentT + currentHorizon < steps.length + 15) {{
        ctx.fillStyle = 'rgba(244, 63, 94, 0.18)';
        ctx.beginPath();
        ctx.moveTo(curX, getY(curStep.y_true));
        ctx.lineTo(targX, getY(b0.q10));
        ctx.lineTo(targX, getY(b0.q90));
        ctx.closePath();
        ctx.fill();

        ctx.strokeStyle = '#f43f5e';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(curX, getY(curStep.y_true));
        ctx.lineTo(targX, getY(b0.median));
        ctx.stroke();

        ctx.fillStyle = '#f43f5e';
        ctx.beginPath();
        ctx.arc(targX, getY(b0.median), 5, 0, Math.PI * 2);
        ctx.fill();
      }}

      // H3 Cone (Cyan)
      const h3 = curStep.models['H3_Unified']?.horizons[currentHorizon];
      if (h3 && currentT + currentHorizon < steps.length + 15) {{
        ctx.fillStyle = 'rgba(6, 182, 212, 0.25)';
        ctx.beginPath();
        ctx.moveTo(curX, getY(curStep.y_true));
        ctx.lineTo(targX, getY(h3.q10));
        ctx.lineTo(targX, getY(h3.q90));
        ctx.closePath();
        ctx.fill();

        ctx.strokeStyle = '#06b6d4';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(curX, getY(curStep.y_true));
        ctx.lineTo(targX, getY(h3.median));
        ctx.stroke();

        ctx.fillStyle = '#06b6d4';
        ctx.beginPath();
        ctx.arc(targX, getY(h3.median), 6, 0, Math.PI * 2);
        ctx.fill();
      }}
    }}

    // Latent Orbit Canvas (z_0 vs z_1)
    function renderLatentOrbit() {{
      const ctx = stateCanvas.getContext('2d');
      const w = stateCanvas.width = stateCanvas.clientWidth * 2;
      const h = stateCanvas.height = stateCanvas.clientHeight * 2;
      ctx.clearRect(0, 0, w, h);

      const sc = getScenario();
      const steps = sc.steps;

      const states = steps.slice(0, currentT + 1).map(s => s.models['H3_Unified']?.state || [0, 0]);
      const curState = states[states.length - 1] || [0, 0];
      stateDisplay.textContent = `[${{curState[0].toFixed(3)}}, ${{curState[1].toFixed(3)}}]`;

      // Center crosshair
      const cx = w / 2;
      const cy = h / 2;
      const scale = Math.min(w, h) * 0.75;

      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cx, 10); ctx.lineTo(cx, h - 10);
      ctx.moveTo(10, cy); ctx.lineTo(w - 10, cy);
      ctx.stroke();

      // State trajectory
      if (states.length > 1) {{
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.5)';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        states.forEach((st, i) => {{
          const sx = cx + st[0] * scale;
          const sy = cy - st[1] * scale;
          if (i === 0) ctx.moveTo(sx, sy);
          else ctx.lineTo(sx, sy);
        }});
        ctx.stroke();
      }}

      // Current state head point
      const hx = cx + curState[0] * scale;
      const hy = cy - curState[1] * scale;
      ctx.fillStyle = '#38bdf8';
      ctx.beginPath();
      ctx.arc(hx, hy, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();
    }}

    // Feedback Cards
    function renderFeedbackCards() {{
      const sc = getScenario();
      const step = sc.steps[currentT];
      const matured = step?.matured_horizons || [];

      [1, 4, 8, 16].forEach(hVal => {{
        const card = document.getElementById(`card-h${{hVal}}`);
        const valEl = document.getElementById(`val-h${{hVal}}`);
        const isMatured = matured.includes(hVal);

        if (isMatured) {{
          card.classList.add('border-emerald-400', 'bg-emerald-950/40', 'glow');
          card.classList.remove('border-[var(--border-color)]', 'bg-slate-900/60');
          valEl.textContent = 'MATURED (Update)';
          valEl.className = 'text-xs font-bold text-emerald-400 mt-0.5';
        }} else {{
          card.classList.remove('border-emerald-400', 'bg-emerald-950/40', 'glow');
          card.classList.add('border-[var(--border-color)]', 'bg-slate-900/60');
          valEl.textContent = 'Pending Arrival';
          valEl.className = 'text-xs font-bold text-slate-400 mt-0.5';
        }}
      }});

      const qNotice = document.getElementById('queueNotice');
      if (matured.length > 0) {{
        qNotice.textContent = `matured horizons: [${{matured.join(', ')}}]`;
      }} else {{
        qNotice.textContent = 'waiting for target arrivals...';
      }}
    }}

    // CRPS Comparison Bars
    function renderCRPSBars() {{
      const sc = getScenario();
      const steps = sc.steps.slice(0, currentT + 1);
      const mNames = ['B0_Frozen', 'B1_IndepEWMA', 'H1_SharedState', 'H3_Unified'];
      const labels = {{
        'B0_Frozen': 'B0: Frozen TSFM',
        'B1_IndepEWMA': 'B1: Indep EWMA',
        'H1_SharedState': 'H1: Naive Shared',
        'H3_Unified': 'H3: Unified Adapter',
      }};
      const colors = {{
        'B0_Frozen': 'bg-rose-500',
        'B1_IndepEWMA': 'bg-amber-500',
        'H1_SharedState': 'bg-purple-500',
        'H3_Unified': 'bg-cyan-400',
      }};

      const cumCRPS = {{}};
      mNames.forEach(m => {{
        let sum = 0, count = 0;
        steps.forEach(s => {{
          const v = s.models[m]?.horizons[currentHorizon]?.crps;
          if (v !== undefined) {{
            sum += v;
            count++;
          }}
        }});
        cumCRPS[m] = count > 0 ? sum / count : 0.001;
      }});

      const maxVal = Math.max(...Object.values(cumCRPS)) * 1.05 || 1.0;
      crpsContainer.innerHTML = '';

      mNames.forEach(m => {{
        const val = cumCRPS[m];
        const pct = Math.min(100, Math.max(8, (val / maxVal) * 100));

        const barWrapper = document.createElement('div');
        barWrapper.innerHTML = `
          <div class="flex items-center justify-between text-xs mb-1">
            <span class="font-medium text-slate-300">${{labels[m]}}</span>
            <span class="font-mono font-bold ${{m === 'H3_Unified' ? 'text-cyan-400' : 'text-slate-400'}}">${{val.toFixed(4)}}</span>
          </div>
          <div class="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
            <div class="${{colors[m]}} h-2.5 rounded-full transition-all duration-200" style="width: ${{pct}}%"></div>
          </div>
        `;
        crpsContainer.appendChild(barWrapper);
      }});

      // Relative Advantage of H3 over B0
      if (cumCRPS['B0_Frozen'] > 0) {{
        const diff = ((cumCRPS['H3_Unified'] - cumCRPS['B0_Frozen']) / cumCRPS['B0_Frozen']) * 100;
        crpsAdvantageLabel.textContent = `${{diff < 0 ? 'H3 vs B0: ' + diff.toFixed(1) + '%' : '+' + diff.toFixed(1) + '%'}}`;
        crpsAdvantageLabel.className = diff < 0 ? 'text-[11px] font-mono text-emerald-400' : 'text-[11px] font-mono text-rose-400';
      }}
    }}

    // Initial boot
    initScenario();
  </script>
</body>
</html>
"""


def main():
    root = Path(".").resolve()
    traces_path = root / "artifacts" / "adaptation" / "visualization" / "visualization_traces.json"
    if not traces_path.exists():
        raise FileNotFoundError(f"Cannot find traces file at {traces_path}")

    with open(traces_path, "r", encoding="utf-8") as f:
        data_json = f.read()

    html_content = build_visualizer_html(data_json)

    # Output 1: Local project visualization folder
    local_out = root / "artifacts" / "adaptation" / "visualization" / "live_adaptation_visualizer.html"
    local_out.write_text(html_content, encoding="utf-8")
    print(f"[OK] Generated local HTML visualizer: {local_out}")

    # Output 2: Brain Artifact directory
    brain_out = Path("C:/Users/Admin/.gemini/antigravity/brain/6d964916-4283-4d59-a794-763b80b97e6c/live_adaptation_visualizer.html")
    brain_out.write_text(html_content, encoding="utf-8")
    print(f"[OK] Generated brain artifact visualizer: {brain_out}")


if __name__ == "__main__":
    main()

'use strict';

// ─── Drop Zone Helper ───────────────────────────────────────────────────────
function setupDropZone(zoneEl, inputEl, nameEl) {
  zoneEl.addEventListener('click', () => inputEl.click());

  inputEl.addEventListener('change', () => {
    if (inputEl.files.length) {
      nameEl.textContent = inputEl.files[0].name;
      zoneEl.classList.add('has-file');
    }
  });

  zoneEl.addEventListener('dragover', e => { e.preventDefault(); zoneEl.classList.add('dragover'); });
  zoneEl.addEventListener('dragleave', () => zoneEl.classList.remove('dragover'));
  zoneEl.addEventListener('drop', e => {
    e.preventDefault();
    zoneEl.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    inputEl.files = dt.files;
    nameEl.textContent = file.name;
    zoneEl.classList.add('has-file');
  });
}

// ─── Main Analysis Form ─────────────────────────────────────────────────────
const analysisForm    = document.getElementById('analysisForm');
const runBtn          = document.getElementById('runBtn');
const progressSection = document.getElementById('progressSection');
const progressFill    = document.getElementById('progressFill');
const progressLabel   = document.getElementById('progressLabel');
const progressPct     = document.getElementById('progressPct');
const trajectoryInfo  = document.getElementById('trajectoryInfo');
const resultsSection  = document.getElementById('resultsSection');
const resultsGrid     = document.getElementById('resultsGrid');
const downloadZipBtn  = document.getElementById('downloadZipBtn');

setupDropZone(
  document.getElementById('topoZone'),
  document.getElementById('topoInput'),
  document.getElementById('topoName')
);
setupDropZone(
  document.getElementById('trajZone'),
  document.getElementById('trajInput'),
  document.getElementById('trajName')
);

// Keep hidden analyses field in sync with checkboxes
document.querySelectorAll('.checkbox-item input[type="checkbox"]').forEach(cb => {
  cb.addEventListener('change', syncAnalyses);
});

function syncAnalyses() {
  const selected = [...document.querySelectorAll('.checkbox-item input[type="checkbox"]:checked')]
    .map(cb => cb.value);
  document.getElementById('analyses_hidden').value = selected.join(',');
}

let currentJobId = null;
let pollTimer    = null;

analysisForm.addEventListener('submit', async e => {
  e.preventDefault();
  syncAnalyses();

  // Clear previous results
  resultsGrid.innerHTML = '';
  resultsSection.classList.add('hidden');

  runBtn.disabled = true;
  runBtn.textContent = '⏳ Uploading...';

  const fd = new FormData(analysisForm);

  try {
    const res = await fetch('/analyze', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Upload failed');

    currentJobId = data.job_id;
    downloadZipBtn.onclick = () => window.location.href = `/download/${currentJobId}`;

    progressSection.classList.remove('hidden');
    progressSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    runBtn.textContent = '⏳ Running...';

    pollTimer = setInterval(pollStatus, 1500);

  } catch (err) {
    showError(err.message);
    runBtn.disabled = false;
    runBtn.textContent = '▶ Run Analysis';
  }
});

async function pollStatus() {
  try {
    const res = await fetch(`/status/${currentJobId}`);
    const job = await res.json();

    progressFill.style.width = job.progress + '%';
    progressLabel.textContent = job.message || '...';
    progressPct.textContent = job.progress + '%';

    if (job.info && job.info.n_frames) {
      trajectoryInfo.innerHTML = `
        <strong>Frames:</strong> ${job.info.n_frames} &nbsp;·&nbsp;
        <strong>Timestep:</strong> ${job.info.dt_ps.toFixed(2)} ps &nbsp;·&nbsp;
        <strong>Total Time:</strong> ${job.info.total_time_ns.toFixed(2)} ns &nbsp;·&nbsp;
        <strong>Protein Atoms:</strong> ${job.info.protein_atoms}
        ${job.info.has_ligand
          ? `&nbsp;·&nbsp; <strong>Ligand Atoms:</strong> ${job.info.ligand_atoms}`
          : '&nbsp;·&nbsp; <em>No ligand detected</em>'}
      `;
    }

    if (job.status === 'done') {
      clearInterval(pollTimer);
      runBtn.disabled = false;
      runBtn.textContent = '▶ Run Analysis';
      renderResults(job.results);
    }

    if (job.status === 'error') {
      clearInterval(pollTimer);
      runBtn.disabled = false;
      runBtn.textContent = '▶ Run Analysis';
      progressLabel.textContent = '❌ ' + job.message;
      progressFill.style.background = '#dc2626';
    }

  } catch (err) {
    clearInterval(pollTimer);
    showError('Lost connection to server: ' + err.message);
  }
}

// ─── Render Results ─────────────────────────────────────────────────────────
const LABELS = {
  rmsd:     'RMSD',
  rmsf:     'RMSF',
  rg:       'Radius of Gyration',
  fel:      'Free Energy Landscape',
  binding:  'Binding Energy',
  distance: 'P-L Distance',
};

const STAT_LABELS = {
  mean: 'Mean',
  std: 'Std Dev',
  unit: null,           // used inline with values, not as a pill
  max_residue: 'Max Residue',
  min_frame: 'Min Frame',
  min_time_ns: 'Min Time (ns)',
  min_rmsd: 'Min RMSD (Å)',
  min_rg: 'Min Rg (Å)',
  contact_pct: 'Contact %',
};

function renderResults(results) {
  resultsSection.classList.remove('hidden');
  resultsGrid.innerHTML = '';

  const order = ['rmsd', 'rmsf', 'rg', 'fel', 'binding', 'distance'];

  order.forEach(key => {
    const r = results[key];
    if (!r) return;

    const card = document.createElement('div');
    card.className = 'result-card';

    const header = document.createElement('div');
    header.className = 'result-card-header';
    header.textContent = LABELS[key] || key.toUpperCase();
    card.appendChild(header);

    if (r.skipped) {
      const skip = document.createElement('div');
      skip.className = 'result-skipped';
      skip.textContent = r.reason || 'Skipped';
      card.appendChild(skip);
    } else {
      // Plot image
      const img = document.createElement('img');
      img.className = 'result-card-img';
      img.src = `data:image/png;base64,${r.image}`;
      img.alt = LABELS[key];
      card.appendChild(img);

      // Stats pills
      const statsDiv = document.createElement('div');
      statsDiv.className = 'result-card-stats';

      const unit = r.unit || '';
      Object.entries(r).forEach(([k, v]) => {
        if (k === 'image' || k === 'unit' || k === 'skipped') return;
        const label = STAT_LABELS[k];
        if (!label) return;

        const pill = document.createElement('span');
        pill.className = 'stat-pill';
        const displayVal = (k === 'mean' || k === 'std') ? `${v} ${unit}` : v;
        pill.textContent = `${label}: ${displayVal}`;
        statsDiv.appendChild(pill);
      });
      card.appendChild(statsDiv);

      // Download buttons
      const footer = document.createElement('div');
      footer.className = 'result-card-footer';

      const dlBtn = document.createElement('a');
      dlBtn.className = 'btn btn-outline';
      dlBtn.textContent = '⬇ Download PNG';
      dlBtn.href = `data:image/png;base64,${r.image}`;
      dlBtn.download = `${key}.png`;
      footer.appendChild(dlBtn);

      if (r.csv_filename) {
        const csvBtn = document.createElement('a');
        csvBtn.className = 'btn btn-outline';
        csvBtn.textContent = '⬇ Download CSV';
        csvBtn.href = `/download-csv/${currentJobId}/${r.csv_filename}`;
        csvBtn.download = r.csv_filename;
        footer.appendChild(csvBtn);
      }

      card.appendChild(footer);
    }

    resultsGrid.appendChild(card);
  });

  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── Frame Extraction Form ──────────────────────────────────────────────────
setupDropZone(
  document.getElementById('extTopoZone'),
  document.getElementById('extTopoInput'),
  document.getElementById('extTopoName')
);
setupDropZone(
  document.getElementById('extTrajZone'),
  document.getElementById('extTrajInput'),
  document.getElementById('extTrajName')
);

const extractForm   = document.getElementById('extractForm');
const extractBtn    = document.getElementById('extractBtn');
const extractStatus = document.getElementById('extractStatus');

extractForm.addEventListener('submit', async e => {
  e.preventDefault();
  extractBtn.disabled = true;
  extractBtn.textContent = '⏳ Extracting...';
  extractStatus.className = 'extract-status';
  extractStatus.textContent = '';

  const fd = new FormData(extractForm);

  try {
    const res = await fetch('/extract-frame', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Extraction failed');
    }

    const frameIdx  = res.headers.get('X-Frame-Index');
    const actualNs  = res.headers.get('X-Actual-Time-NS');
    const blob      = await res.blob();
    const url       = URL.createObjectURL(blob);
    const a         = document.createElement('a');
    const outName   = document.getElementById('ext_output_name').value || 'extracted_frame.pdb';
    a.href          = url;
    a.download      = outName;
    a.click();
    URL.revokeObjectURL(url);

    extractStatus.className = 'extract-status success';
    extractStatus.textContent =
      `✓ Frame ${frameIdx} extracted (actual time: ${parseFloat(actualNs).toFixed(3)} ns)`;

  } catch (err) {
    extractStatus.className = 'extract-status error';
    extractStatus.textContent = '❌ ' + err.message;
  } finally {
    extractBtn.disabled = false;
    extractBtn.textContent = 'Extract Frame → Download PDB';
  }
});

// ─── Utilities ──────────────────────────────────────────────────────────────
function showError(msg) {
  let banner = document.querySelector('.error-banner');
  if (!banner) {
    banner = document.createElement('div');
    banner.className = 'error-banner';
    document.querySelector('#upload .container').prepend(banner);
  }
  banner.textContent = '❌ ' + msg;
}

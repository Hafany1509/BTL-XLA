const $ = (id) => document.getElementById(id);
const MAX_BYTES = 10 * 1024 * 1024;
let currentFile = null;
let models = [];

const kb = (n) => `${(n / 1024).toLocaleString("vi-VN", { maximumFractionDigits: 1 })} KB`;
const score = (x) =>
  x < 1e-4 ? "< 0,0001" : x > 0.9999 ? "> 0,9999"
    : x.toLocaleString("vi-VN", { minimumFractionDigits: 4, maximumFractionDigits: 4 });

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}

function updateRunButton() {
  const anyModel = document.querySelectorAll("#model-list input:checked").length > 0;
  $("run").disabled = !(currentFile && anyModel);
}

async function loadStatus() {
  try {
    const health = await (await fetch("/api/health")).json();
    models = health.models;
    const ready = models.filter((m) => m.available).length;
    $("status").textContent = `máy chủ sẵn sàng, ${ready}/${models.length} mô hình`;
    $("status").className = `status ${health.status === "ok" ? "ok" : "bad"}`;
    renderModels();
  } catch {
    $("status").textContent = "không kết nối được máy chủ";
    $("status").className = "status bad";
  }
}

function renderModels() {
  const list = $("model-list");
  list.innerHTML = "";
  for (const m of models) {
    const label = document.createElement("label");
    if (!m.available) label.className = "off";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = m.key;
    input.disabled = !m.available;
    input.checked = m.available && (m.key === "baseline" || m.key === "jpeg");
    input.addEventListener("change", updateRunButton);
    const text = document.createElement("span");
    const desc = document.createElement("small");
    text.textContent = m.name;
    desc.textContent = m.available ? m.description : m.error || "Không khả dụng";
    text.append(desc);
    label.append(input, text);
    list.append(label);
  }
  updateRunButton();
}

async function loadSamples() {
  try {
    const samples = await (await fetch("/api/samples")).json();
    for (const s of samples) {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      const img = document.createElement("img");
      img.src = s.url;
      img.alt = "";
      button.append(img, s.label);
      button.addEventListener("click", async () => {
        const blob = await (await fetch(s.url)).blob();
        setFile(new File([blob], s.file, { type: blob.type }), s.label);
      });
      item.append(button);
      $("sample-list").append(item);
    }
  } catch {
    /* samples are optional */
  }
}

function setFile(file, note = "") {
  showError("");
  if (!["image/png", "image/jpeg"].includes(file.type)) {
    showError("Chỉ nhận ảnh JPG hoặc PNG.");
    return;
  }
  if (file.size > MAX_BYTES) {
    showError("Ảnh lớn hơn 10 MB.");
    return;
  }
  currentFile = file;
  const preview = $("preview");
  if (preview.src) URL.revokeObjectURL(preview.src);
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  $("drop-hint").hidden = true;
  $("file-info").textContent = `${file.name}, ${kb(file.size)}${note ? ". " + note : ""}`;
  updateRunButton();
}

function renderResults(data) {
  const table = document.createElement("table");
  table.className = "res";
  table.innerHTML = `<thead><tr><th>Mô hình</th><th>Kết luận</th><th class="num">fake_score</th><th style="width:38%"></th></tr></thead><tbody></tbody>`;
  const body = table.querySelector("tbody");
  for (const r of data.results) {
    const fake = r.label === "FAKE";
    const row = document.createElement("tr");
    row.innerHTML = `
      <td><span class="model-name"></span><span class="model-sub"></span></td>
      <td><span class="verdict ${fake ? "fake" : "real"}">${fake ? "ảnh AI" : "ảnh thật"}</span></td>
      <td class="num"></td>
      <td><div class="track" title="vạch đứng: ngưỡng 0,5"><div class="fill ${fake ? "fake" : "real"}"></div><div class="cut"></div></div></td>`;
    row.querySelector(".model-name").textContent = r.name;
    row.querySelector(".model-sub").textContent = `${r.checkpoint_sha256.slice(0, 12)} · ${Math.round(r.inference_time_ms)} ms`;
    row.querySelector(".num").textContent = score(r.fake_score);
    row.querySelector(".fill").style.width = `${Math.max(r.fake_score * 100, 0.5)}%`;
    body.append(row);
  }
  $("results").replaceChildren(table);

  let meta = `Đầu vào: ${data.format} ${data.width}×${data.height}, ${kb(data.file_bytes)}.`;
  if (data.jpeg_quality) meta += ` Đã nén lại JPEG q${data.jpeg_quality} (còn ${kb(data.compressed_bytes)}) trước khi đưa vào mô hình.`;
  $("meta").textContent = meta;
}

async function run() {
  showError("");
  const chosen = [...document.querySelectorAll("#model-list input:checked")].map((i) => i.value);
  const form = new FormData();
  form.append("image", currentFile);
  form.append("models_csv", chosen.join(","));
  form.append("jpeg_quality", document.querySelector("input[name=quality]:checked").value);

  const button = $("run");
  button.disabled = true;
  button.textContent = "Đang chạy…";
  try {
    const response = await fetch("/api/predict", { method: "POST", body: form });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || `Máy chủ trả lỗi ${response.status}`);
    renderResults(data);
  } catch (error) {
    showError(error.message === "Failed to fetch" ? "Không kết nối được máy chủ." : error.message);
  } finally {
    button.textContent = "Chạy";
    updateRunButton();
  }
}

const dropzone = $("dropzone");
$("file").addEventListener("change", (e) => e.target.files[0] && setFile(e.target.files[0]));
for (const type of ["dragenter", "dragover"]) {
  dropzone.addEventListener(type, (e) => { e.preventDefault(); dropzone.classList.add("drag"); });
}
for (const type of ["dragleave", "drop"]) {
  dropzone.addEventListener(type, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); });
}
dropzone.addEventListener("drop", (e) => e.dataTransfer.files[0] && setFile(e.dataTransfer.files[0]));
$("run").addEventListener("click", run);

loadStatus();
loadSamples();

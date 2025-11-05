// app_quantile.js — apply QuantileTransformer (uniform->normal) in browser
let sess, meta;

function lower(s){ return (s||"").toString().trim().toLowerCase(); }
function clamp01(p){ return Math.min(1-1e-12, Math.max(1e-12, p)); }

// Φ^{-1}(p): inverse normal CDF (probit) — rational approximation (Beasley-Springer/Moro style)
function probit(p){
  // Abramowitz & Stegun style approximation
  const a1=-39.6968302866538, a2=220.946098424521, a3=-275.928510446969,
        a4=138.357751867269, a5=-30.6647980661472, a6=2.50662827745924;
  const b1=-54.4760987982241, b2=161.585836858041, b3=-155.698979859887,
        b4=66.8013118877197, b5=-13.2806815528857;
  const c1=-7.78489400243029e-03, c2=-0.322396458041136, c3=-2.40075827716184,
        c4=-2.54973253934373, c5=4.37466414146497, c6=2.93816398269878;
  const d1=7.78469570904146e-03, d2=0.32246712907004, d3=2.445134137143,
        d4=3.75440866190742;
  const plow=0.02425, phigh=1-plow;
  let q, r;
  if (p < plow){
    q = Math.sqrt(-2*Math.log(p));
    return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
           ((((d1*q+d2)*q+d3)*q+d4)*q+1);
  }
  if (phigh < p){
    q = Math.sqrt(-2*Math.log(1-p));
    return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
             ((((d1*q+d2)*q+d3)*q+d4)*q+1);
  }
  q = p-0.5; r = q*q;
  return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q /
         (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1);
}

// Given a value v and the quantiles array q (length nQ), compute uniform CDF via piecewise linear interpolation
function cdfFromQuantiles(v, q){
  const n = q.length;
  if (v <= q[0]) return 0.0;
  if (v >= q[n-1]) return 1.0;
  // binary search for interval
  let lo=0, hi=n-1;
  while (hi-lo>1){
    const mid = (lo+hi)>>1;
    if (v < q[mid]) hi=mid; else lo=mid;
  }
  const t = (v - q[lo]) / Math.max(1e-12, (q[hi]-q[lo]));
  const p_lo = lo/(n-1), p_hi = hi/(n-1);
  return p_lo + (p_hi - p_lo) * t;
}

async function loadAll(){
  meta = await (await fetch('./meta_quantile.json')).json();
  sess = await ort.InferenceSession.create('./gplay_loginstalls_quantile.onnx');
  log('Loaded model + quantiles.');
}
function log(msg){
  const el = document.getElementById('out');
  el.textContent = (el.textContent ? el.textContent + '\n' : '') + msg;
}

function buildVector(){
  const feats = meta.feature_cols_order;
  const x = new Float32Array(feats.length); // zeros

  const cat = lower(document.getElementById('category').value);
  const cr  = lower(document.getElementById('contentRating').value);
  const gp  = lower(document.getElementById('genre').value);
  const reviews = parseFloat(document.getElementById('reviews').value);
  const rating  = parseFloat(document.getElementById('rating').value);
  const sizeMB  = parseFloat(document.getElementById('size').value);
  const price   = parseFloat(document.getElementById('price').value);
  const typeSel = lower(document.getElementById('type').value);

  const put = (name, val)=>{
    const i = feats.indexOf(name);
    if(i>=0) x[i] = Number.isFinite(val)? val : NaN;
  };
  put('Price_num', price);
  put('Size_MB',   sizeMB);
  put('Reviews_num', reviews);
  put('Rating_num',  rating);
  put('Type_is_paid', typeSel==='paid'? 1 : 0);
  put('days_since_update', NaN); // imputed below

  const hot = (prefix, level)=>{
    const col = `${prefix}${level}`;
    let idx = feats.indexOf(col);
    if(idx>=0){ x[idx] = 1; return; }
    const other = `${prefix}other`;
    idx = feats.indexOf(other);
    if(idx>=0) x[idx] = 1;
  };
  hot('Category_cat_',      cat);
  hot('ContentRating_cat_', cr);
  hot('Genre_primary_',     gp);

  // --- Numeric impute ---
  const numIdx = meta.num_col_idx;
  const meds = meta.train_numeric_medians;
  for(let k=0;k<numIdx.length;k++){
    const j = numIdx[k];
    if(!Number.isFinite(x[j])) x[j] = meds[k];
  }

  // --- Quantile -> (uniform) -> normal (probit) ---
  const Q = meta.quantiles;           // [nQ][nNum]
  const nQ = meta.n_quantiles|0;
  for(let k=0;k<numIdx.length;k++){
    const j = numIdx[k];
    // collect k-th feature’s quantiles
    const q = new Array(nQ);
    for(let i=0;i<nQ;i++){ q[i] = Q[i][k]; }
    // uniform CDF in [0,1]
    const pu = cdfFromQuantiles(x[j], q);
    // to normal
    const p = clamp01(pu);
    x[j] = probit(p);
  }

  // Safety: replace any remaining non-finite
  for(let i=0;i<x.length;i++){
    if(!Number.isFinite(x[i])) x[i] = 0.0;
  }
  return x;
}

function toBucket(v){
  const bins = meta.bins;
  let best = bins[0], bestd = Math.abs(v - bins[0]);
  for(const b of bins){
    const d = Math.abs(v-b);
    if(d<bestd){ best=b; bestd=d; }
  }
  return best;
}

async function predict(){
  try{
    const x = buildVector();
    const feeds = { input: new ort.Tensor('float32', x, [1, x.length]) };
    const out = await sess.run(feeds);
    const ylog = out.output.data[0];
    const installs = Math.expm1(ylog);
    const bucket = toBucket(installs);
    document.getElementById('out').textContent =
      `Predicted installs: ${Math.round(installs).toLocaleString()}  (nearest bucket: ${bucket.toLocaleString()}+)`;
  }catch(err){
    document.getElementById('out').textContent = 'Error: ' + err;
    console.error(err);
  }
}

document.getElementById('predict').addEventListener('click', predict);
loadAll();

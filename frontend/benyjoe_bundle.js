/* BENY-JOE CINE IA PRO v5.0 - Animation & UI Library */
(function(global){'use strict';
const BJ = global.BenyJoe = global.BenyJoe || {};
BJ.VERSION = '5.0.0';
BJ.STYLES = ['cinematic','cyberpunk','epic_battle','nature','scifi','noir','fantasy','horror','romantique'];
BJ.VOICES = ['masculin','feminin','dramatique','jeune','epique'];
BJ.MUSIC  = ['cinematique','ambiante','cyberpunk','nature','scifi','noir','fantasy','horror','epic_battle','romantique'];

// Easing functions
BJ.ease = {
  linear:      t => t,
  quad:        t => t*t,
  cubic:       t => t*t*t,
  quart:       t => t*t*t*t,
  sine:        t => 1 - Math.cos(t * Math.PI / 2),
  expo:        t => t === 0 ? 0 : Math.pow(2, 10 * (t - 1)),
  circ:        t => 1 - Math.sqrt(1 - t * t),
  back:        t => t * t * (2.7 * t - 1.7),
  elastic:     t => t === 0 ? 0 : t === 1 ? 1 : -Math.pow(2, 10*(t-1)) * Math.sin((t-1.1)*5*Math.PI),
  bounce: t => {
    if(t < 1/2.75) return 7.5625*t*t;
    if(t < 2/2.75) return 7.5625*(t-=1.5/2.75)*t+0.75;
    if(t < 2.5/2.75) return 7.5625*(t-=2.25/2.75)*t+0.9375;
    return 7.5625*(t-=2.625/2.75)*t+0.984375;
  },
};

// Animate function
BJ.animate = (opts) => {
  const {duration=400, from=0, to=1, easing='linear', onUpdate, onComplete} = opts;
  const easeFn = BJ.ease[easing] || BJ.ease.linear;
  let start = null;
  const step = (timestamp) => {
    if(!start) start = timestamp;
    const progress = Math.min((timestamp - start) / duration, 1);
    const value = from + (to - from) * easeFn(progress);
    if(onUpdate) onUpdate(value, progress);
    if(progress < 1) requestAnimationFrame(step);
    else if(onComplete) onComplete();
  };
  requestAnimationFrame(step);
};

// Progress bar
BJ.ProgressBar = class {
  constructor(el) { this.el = typeof el === 'string' ? document.getElementById(el) : el; this.pct = 0; }
  set(pct, step='') {
    this.pct = Math.max(0, Math.min(100, pct));
    if(!this.el) return;
    this.el.innerHTML = '<div style="display:flex;justify-content:space-between;font-size:12px;color:#8888aa;margin-bottom:6px"><span>' + step + '</span><span>' + Math.round(this.pct) + '%</span></div><div style="background:#1a1a28;border-radius:20px;height:8px;overflow:hidden"><div style="height:100%;width:' + this.pct + '%;background:linear-gradient(90deg,#7c5cbf,#5b8dee);border-radius:20px;transition:width .4s"></div></div>';
  }
  animate(from, to, duration=400) {
    BJ.animate({ from, to, duration, easing:'quad', onUpdate: (v) => this.set(v) });
  }
};

// Toast
BJ.toast = (msg, type='ok', ms=4000) => {
  let wrap = document.getElementById('_bjt');
  if(!wrap){wrap=document.createElement('div');wrap.id='_bjt';Object.assign(wrap.style,{position:'fixed',bottom:'20px',right:'20px',zIndex:'9999',display:'flex',flexDirection:'column',gap:'8px'});document.body.appendChild(wrap);}
  const el=document.createElement('div');
  el.textContent=msg;
  Object.assign(el.style,{background:type==='err'?'#2a1020':'#102a20',border:'1px solid '+(type==='err'?'#e84b6e':'#4be8a0'),color:type==='err'?'#e84b6e':'#4be8a0',borderRadius:'10px',padding:'12px 18px',fontSize:'13px',maxWidth:'300px',boxShadow:'0 4px 20px rgba(0,0,0,.5)'});
  wrap.appendChild(el);setTimeout(()=>el.remove(),ms);
};

// API Client
BJ.api = {
  base: '',
  call: async (method, path, data) => {
    const opts = { method, headers: {'Content-Type':'application/json'} };
    if(data) opts.body = JSON.stringify(data);
    const r = await fetch(BJ.api.base + path, opts);
    return r.json();
  },
  auth:      (pin) => BJ.api.call('POST','/api/auth',{pin}),
  health:    () => BJ.api.call('GET','/api/health'),
  generate:  (p) => BJ.api.call('POST','/api/generate',p),
  image:     (p) => BJ.api.call('POST','/api/generate_image',p),
  img2video: (p) => BJ.api.call('POST','/api/img2video',p),
  status:    (id) => BJ.api.call('GET','/api/status/'+id),
  cancel:    (id) => BJ.api.call('POST','/api/cancel/'+id),
  gpuUrl:    (url) => BJ.api.call('POST','/api/gpu_url',{url}),
  history:   () => BJ.api.call('GET','/api/history'),
};

// Poller
BJ.Poller = class {
  constructor(jobId, {interval=4000, onUpdate, onDone, onError}={}) {
    this.jobId=jobId; this.interval=interval;
    this.onUpdate=onUpdate||(() => {}); this.onDone=onDone||(() => {}); this.onError=onError||(() => {});
    this._t=null;
  }
  start() { this._t=setInterval(()=>this._poll(),this.interval); this._poll(); return this; }
  stop()  { clearInterval(this._t); }
  async _poll() {
    try {
      const d = await BJ.api.status(this.jobId);
      this.onUpdate(d);
      if(d.status==='done'){this.stop();this.onDone(d.result);}
      else if(['error','cancelled'].includes(d.status)){this.stop();this.onError(d.error||d.status);}
    } catch(e){ console.warn('[Poller]',e); }
  }
};

// Utils
BJ.utils = {
  toBase64: f => new Promise((res,rej)=>{const r=new FileReader();r.onload=e=>res(e.target.result.split(',')[1]);r.onerror=rej;r.readAsDataURL(f);}),
  sleep: ms => new Promise(r=>setTimeout(r,ms)),
  debounce: (fn,ms) => {let t;return (...a)=>{clearTimeout(t);t=setTimeout(()=>fn(...a),ms);};},
  pick: arr => arr[Math.floor(Math.random()*arr.length)],
  formatDur: s => s<60?Math.round(s)+'s':(Math.floor(s/60)+'m'+Math.round(s%60)+'s'),
  formatMB: b => (b/1048576).toFixed(1)+' MB',
};

})(window);/* BENY-JOE CINE IA PRO v5.0 - Animation & UI Library */
(function(global){'use strict';
const BJ = global.BenyJoe = global.BenyJoe || {};
BJ.VERSION = '5.0.0';
BJ.STYLES = ['cinematic','cyberpunk','epic_battle','nature','scifi','noir','fantasy','horror','romantique'];
BJ.VOICES = ['masculin','feminin','dramatique','jeune','epique'];
BJ.MUSIC  = ['cinematique','ambiante','cyberpunk','nature','scifi','noir','fantasy','horror','epic_battle','romantique'];

// Easing functions
BJ.ease = {
  linear:      t => t,
  quad:        t => t*t,
  cubic:       t => t*t*t,
  quart:       t => t*t*t*t,
  sine:        t => 1 - Math.cos(t * Math.PI / 2),
  expo:        t => t === 0 ? 0 : Math.pow(2, 10 * (t - 1)),
  circ:        t => 1 - Math.sqrt(1 - t * t),
  back:        t => t * t * (2.7 * t - 1.7),
  elastic:     t => t === 0 ? 0 : t === 1 ? 1 : -Math.pow(2, 10*(t-1)) * Math.sin((t-1.1)*5*Math.PI),
  bounce: t => {
    if(t < 1/2.75) return 7.5625*t*t;
    if(t < 2/2.75) return 7.5625*(t-=1.5/2.75)*t+0.75;
    if(t < 2.5/2.75) return 7.5625*(t-=2.25/2.75)*t+0.9375;
    return 7.5625*(t-=2.625/2.75)*t+0.984375;
  },
};

// Animate function
BJ.animate = (opts) => {
  const {duration=400, from=0, to=1, easing='linear', onUpdate, onComplete} = opts;
  const easeFn = BJ.ease[easing] || BJ.ease.linear;
  let start = null;
  const step = (timestamp) => {
    if(!start) start = timestamp;
    const progress = Math.min((timestamp - start) / duration, 1);
    const value = from + (to - from) * easeFn(progress);
    if(onUpdate) onUpdate(value, progress);
    if(progress < 1) requestAnimationFrame(step);
    else if(onComplete) onComplete();
  };
  requestAnimationFrame(step);
};

// Progress bar
BJ.ProgressBar = class {
  constructor(el) { this.el = typeof el === 'string' ? document.getElementById(el) : el; this.pct = 0; }
  set(pct, step='') {
    this.pct = Math.max(0, Math.min(100, pct));
    if(!this.el) return;
    this.el.innerHTML = '<div style="display:flex;justify-content:space-between;font-size:12px;color:#8888aa;margin-bottom:6px"><span>' + step + '</span><span>' + Math.round(this.pct) + '%</span></div><div style="background:#1a1a28;border-radius:20px;height:8px;overflow:hidden"><div style="height:100%;width:' + this.pct + '%;background:linear-gradient(90deg,#7c5cbf,#5b8dee);border-radius:20px;transition:width .4s"></div></div>';
  }
  animate(from, to, duration=400) {
    BJ.animate({ from, to, duration, easing:'quad', onUpdate: (v) => this.set(v) });
  }
};

// Toast
BJ.toast = (msg, type='ok', ms=4000) => {
  let wrap = document.getElementById('_bjt');
  if(!wrap){wrap=document.createElement('div');wrap.id='_bjt';Object.assign(wrap.style,{position:'fixed',bottom:'20px',right:'20px',zIndex:'9999',display:'flex',flexDirection:'column',gap:'8px'});document.body.appendChild(wrap);}
  const el=document.createElement('div');
  el.textContent=msg;
  Object.assign(el.style,{background:type==='err'?'#2a1020':'#102a20',border:'1px solid '+(type==='err'?'#e84b6e':'#4be8a0'),color:type==='err'?'#e84b6e':'#4be8a0',borderRadius:'10px',padding:'12px 18px',fontSize:'13px',maxWidth:'300px',boxShadow:'0 4px 20px rgba(0,0,0,.5)'});
  wrap.appendChild(el);setTimeout(()=>el.remove(),ms);
};

// API Client
BJ.api = {
  base: '',
  call: async (method, path, data) => {
    const opts = { method, headers: {'Content-Type':'application/json'} };
    if(data) opts.body = JSON.stringify(data);
    const r = await fetch(BJ.api.base + path, opts);
    return r.json();
  },
  auth:      (pin) => BJ.api.call('POST','/api/auth',{pin}),
  health:    () => BJ.api.call('GET','/api/health'),
  generate:  (p) => BJ.api.call('POST','/api/generate',p),
  image:     (p) => BJ.api.call('POST','/api/generate_image',p),
  img2video: (p) => BJ.api.call('POST','/api/img2video',p),
  status:    (id) => BJ.api.call('GET','/api/status/'+id),
  cancel:    (id) => BJ.api.call('POST','/api/cancel/'+id),
  gpuUrl:    (url) => BJ.api.call('POST','/api/gpu_url',{url}),
  history:   () => BJ.api.call('GET','/api/history'),
};

// Poller
BJ.Poller = class {
  constructor(jobId, {interval=4000, onUpdate, onDone, onError}={}) {
    this.jobId=jobId; this.interval=interval;
    this.onUpdate=onUpdate||(() => {}); this.onDone=onDone||(() => {}); this.onError=onError||(() => {});
    this._t=null;
  }
  start() { this._t=setInterval(()=>this._poll(),this.interval); this._poll(); return this; }
  stop()  { clearInterval(this._t); }
  async _poll() {
    try {
      const d = await BJ.api.status(this.jobId);
      this.onUpdate(d);
      if(d.status==='done'){this.stop();this.onDone(d.result);}
      else if(['error','cancelled'].includes(d.status)){this.stop();this.onError(d.error||d.status);}
    } catch(e){ console.warn('[Poller]',e); }
  }
};

// Utils
BJ.utils = {
  toBase64: f => new Promise((res,rej)=>{const r=new FileReader();r.onload=e=>res(e.target.result.split(',')[1]);r.onerror=rej;r.readAsDataURL(f);}),
  sleep: ms => new Promise(r=>setTimeout(r,ms)),
  debounce: (fn,ms) => {let t;return (...a)=>{clearTimeout(t);t=setTimeout(()=>fn(...a),ms);};},
  pick: arr => arr[Math.floor(Math.random()*arr.length)],
  formatDur: s => s<60?Math.round(s)+'s':(Math.floor(s/60)+'m'+Math.round(s%60)+'s'),
  formatMB: b => (b/1048576).toFixed(1)+' MB',
};

})(window);/* BENY-JOE CINE IA PRO v5.0 - Animation & UI Library */
(function(global){'use strict';
const BJ = global.BenyJoe = global.BenyJoe || {};
BJ.VERSION = '5.0.0';
BJ.STYLES = ['cinematic','cyberpunk','epic_battle','nature','scifi','noir','fantasy','horror','romantique'];
BJ.VOICES = ['masculin','feminin','dramatique','jeune','epique'];
BJ.MUSIC  = ['cinematique','ambiante','cyberpunk','nature','scifi','noir','fantasy','horror','epic_battle','romantique'];

// Easing functions
BJ.ease = {
  linear:      t => t,
  quad:        t => t*t,
  cubic:       t => t*t*t,
  quart:       t => t*t*t*t,
  sine:        t => 1 - Math.cos(t * Math.PI / 2),
  expo:        t => t === 0 ? 0 : Math.pow(2, 10 * (t - 1)),
  circ:        t => 1 - Math.sqrt(1 - t * t),
  back:        t => t * t * (2.7 * t - 1.7),
  elastic:     t => t === 0 ? 0 : t === 1 ? 1 : -Math.pow(2, 10*(t-1)) * Math.sin((t-1.1)*5*Math.PI),
  bounce: t => {
    if(t < 1/2.75) return 7.5625*t*t;
    if(t < 2/2.75) return 7.5625*(t-=1.5/2.75)*t+0.75;
    if(t < 2.5/2.75) return 7.5625*(t-=2.25/2.75)*t+0.9375;
    return 7.5625*(t-=2.625/2.75)*t+0.984375;
  },
};

// Animate function
BJ.animate = (opts) => {
  const {duration=400, from=0, to=1, easing='linear', onUpdate, onComplete} = opts;
  const easeFn = BJ.ease[easing] || BJ.ease.linear;
  let start = null;
  const step = (timestamp) => {
    if(!start) start = timestamp;
    const progress = Math.min((timestamp - start) / duration, 1);
    const value = from + (to - from) * easeFn(progress);
    if(onUpdate) onUpdate(value, progress);
    if(progress < 1) requestAnimationFrame(step);
    else if(onComplete) onComplete();
  };
  requestAnimationFrame(step);
};

// Progress bar
BJ.ProgressBar = class {
  constructor(el) { this.el = typeof el === 'string' ? document.getElementById(el) : el; this.pct = 0; }
  set(pct, step='') {
    this.pct = Math.max(0, Math.min(100, pct));
    if(!this.el) return;
    this.el.innerHTML = '<div style="display:flex;justify-content:space-between;font-size:12px;color:#8888aa;margin-bottom:6px"><span>' + step + '</span><span>' + Math.round(this.pct) + '%</span></div><div style="background:#1a1a28;border-radius:20px;height:8px;overflow:hidden"><div style="height:100%;width:' + this.pct + '%;background:linear-gradient(90deg,#7c5cbf,#5b8dee);border-radius:20px;transition:width .4s"></div></div>';
  }
  animate(from, to, duration=400) {
    BJ.animate({ from, to, duration, easing:'quad', onUpdate: (v) => this.set(v) });
  }
};

// Toast
BJ.toast = (msg, type='ok', ms=4000) => {
  let wrap = document.getElementById('_bjt');
  if(!wrap){wrap=document.createElement('div');wrap.id='_bjt';Object.assign(wrap.style,{position:'fixed',bottom:'20px',right:'20px',zIndex:'9999',display:'flex',flexDirection:'column',gap:'8px'});document.body.appendChild(wrap);}
  const el=document.createElement('div');
  el.textContent=msg;
  Object.assign(el.style,{background:type==='err'?'#2a1020':'#102a20',border:'1px solid '+(type==='err'?'#e84b6e':'#4be8a0'),color:type==='err'?'#e84b6e':'#4be8a0',borderRadius:'10px',padding:'12px 18px',fontSize:'13px',maxWidth:'300px',boxShadow:'0 4px 20px rgba(0,0,0,.5)'});
  wrap.appendChild(el);setTimeout(()=>el.remove(),ms);
};

// API Client
BJ.api = {
  base: '',
  call: async (method, path, data) => {
    const opts = { method, headers: {'Content-Type':'application/json'} };
    if(data) opts.body = JSON.stringify(data);
    const r = await fetch(BJ.api.base + path, opts);
    return r.json();
  },
  auth:      (pin) => BJ.api.call('POST','/api/auth',{pin}),
  health:    () => BJ.api.call('GET','/api/health'),
  generate:  (p) => BJ.api.call('POST','/api/generate',p),
  image:     (p) => BJ.api.call('POST','/api/generate_image',p),
  img2video: (p) => BJ.api.call('POST','/api/img2video',p),
  status:    (id) => BJ.api.call('GET','/api/status/'+id),
  cancel:    (id) => BJ.api.call('POST','/api/cancel/'+id),
  gpuUrl:    (url) => BJ.api.call('POST','/api/gpu_url',{url}),
  history:   () => BJ.api.call('GET','/api/history'),
};

// Poller
BJ.Poller = class {
  constructor(jobId, {interval=4000, onUpdate, onDone, onError}={}) {
    this.jobId=jobId; this.interval=interval;
    this.onUpdate=onUpdate||(() => {}); this.onDone=onDone||(() => {}); this.onError=onError||(() => {});
    this._t=null;
  }
  start() { this._t=setInterval(()=>this._poll(),this.interval); this._poll(); return this; }
  stop()  { clearInterval(this._t); }
  async _poll() {
    try {
      const d = await BJ.api.status(this.jobId);
      this.onUpdate(d);
      if(d.status==='done'){this.stop();this.onDone(d.result);}
      else if(['error','cancelled'].includes(d.status)){this.stop();this.onError(d.error||d.status);}
    } catch(e){ console.warn('[Poller]',e); }
  }
};

// Utils
BJ.utils = {
  toBase64: f => new Promise((res,rej)=>{const r=new FileReader();r.onload=e=>res(e.target.result.split(',')[1]);r.onerror=rej;r.readAsDataURL(f);}),
  sleep: ms => new Promise(r=>setTimeout(r,ms)),
  debounce: (fn,ms) => {let t;return (...a)=>{clearTimeout(t);t=setTimeout(()=>fn(...a),ms);};},
  pick: arr => arr[Math.floor(Math.random()*arr.length)],
  formatDur: s => s<60?Math.round(s)+'s':(Math.floor(s/60)+'m'+Math.round(s%60)+'s'),
  formatMB: b => (b/1048576).toFixed(1)+' MB',
};

})(window);/* BENY-JOE CINE IA PRO v5.0 - Animation & UI Library */
(function(global){'use strict';
const BJ = global.BenyJoe = global.BenyJoe || {};
BJ.VERSION = '5.0.0';
BJ.STYLES = ['cinematic','cyberpunk','epic_battle','nature','scifi','noir','fantasy','horror','romantique'];
BJ.VOICES = ['masculin','feminin','dramatique','jeune','epique'];
BJ.MUSIC  = ['cinematique','ambiante','cyberpunk','nature','scifi','noir','fantasy','horror','epic_battle','romantique'];

// Easing functions
BJ.ease = {
  linear:      t => t,
  quad:        t => t*t,
  cubic:       t => t*t*t,
  quart:       t => t*t*t*t,
  sine:        t => 1 - Math.cos(t * Math.PI / 2),
  expo:        t => t === 0 ? 0 : Math.pow(2, 10 * (t - 1)),
  circ:        t => 1 - Math.sqrt(1 - t * t),
  back:        t => t * t * (2.7 * t - 1.7),
  elastic:     t => t === 0 ? 0 : t === 1 ? 1 : -Math.pow(2, 10*(t-1)) * Math.sin((t-1.1)*5*Math.PI),
  bounce: t => {
    if(t < 1/2.75) return 7.5625*t*t;
    if(t < 2/2.75) return 7.5625*(t-=1.5/2.75)*t+0.75;
    if(t < 2.5/2.75) return 7.5625*(t-=2.25/2.75)*t+0.9375;
    return 7.5625*(t-=2.625/2.75)*t+0.984375;
  },
};

// Animate function
BJ.animate = (opts) => {
  const {duration=400, from=0, to=1, easing='linear', onUpdate, onComplete} = opts;
  const easeFn = BJ.ease[easing] || BJ.ease.linear;
  let start = null;
  const step = (timestamp) => {
    if(!start) start = timestamp;
    const progress = Math.min((timestamp - start) / duration, 1);
    const value = from + (to - from) * easeFn(progress);
    if(onUpdate) onUpdate(value, progress);
    if(progress < 1) requestAnimationFrame(step);
    else if(onComplete) onComplete();
  };
  requestAnimationFrame(step);
};

// Progress bar
BJ.ProgressBar = class {
  constructor(el) { this.el = typeof el === 'string' ? document.getElementById(el) : el; this.pct = 0; }
  set(pct, step='') {
    this.pct = Math.max(0, Math.min(100, pct));
    if(!this.el) return;
    this.el.innerHTML = '<div style="display:flex;justify-content:space-between;font-size:12px;color:#8888aa;margin-bottom:6px"><span>' + step + '</span><span>' + Math.round(this.pct) + '%</span></div><div style="background:#1a1a28;border-radius:20px;height:8px;overflow:hidden"><div style="height:100%;width:' + this.pct + '%;background:linear-gradient(90deg,#7c5cbf,#5b8dee);border-radius:20px;transition:width .4s"></div></div>';
  }
  animate(from, to, duration=400) {
    BJ.animate({ from, to, duration, easing:'quad', onUpdate: (v) => this.set(v) });
  }
};

// Toast
BJ.toast = (msg, type='ok', ms=4000) => {
  let wrap = document.getElementById('_bjt');
  if(!wrap){wrap=document.createElement('div');wrap.id='_bjt';Object.assign(wrap.style,{position:'fixed',bottom:'20px',right:'20px',zIndex:'9999',display:'flex',flexDirection:'column',gap:'8px'});document.body.appendChild(wrap);}
  const el=document.createElement('div');
  el.textContent=msg;
  Object.assign(el.style,{background:type==='err'?'#2a1020':'#102a20',border:'1px solid '+(type==='err'?'#e84b6e':'#4be8a0'),color:type==='err'?'#e84b6e':'#4be8a0',borderRadius:'10px',padding:'12px 18px',fontSize:'13px',maxWidth:'300px',boxShadow:'0 4px 20px rgba(0,0,0,.5)'});
  wrap.appendChild(el);setTimeout(()=>el.remove(),ms);
};

// API Client
BJ.api = {
  base: '',
  call: async (method, path, data) => {
    const opts = { method, headers: {'Content-Type':'application/json'} };
    if(data) opts.body = JSON.stringify(data);
    const r = await fetch(BJ.api.base + path, opts);
    return r.json();
  },
  auth:      (pin) => BJ.api.call('POST','/api/auth',{pin}),
  health:    () => BJ.api.call('GET','/api/health'),
  generate:  (p) => BJ.api.call('POST','/api/generate',p),
  image:     (p) => BJ.api.call('POST','/api/generate_image',p),
  img2video: (p) => BJ.api.call('POST','/api/img2video',p),
  status:    (id) => BJ.api.call('GET','/api/status/'+id),
  cancel:    (id) => BJ.api.call('POST','/api/cancel/'+id),
  gpuUrl:    (url) => BJ.api.call('POST','/api/gpu_url',{url}),
  history:   () => BJ.api.call('GET','/api/history'),
};

// Poller
BJ.Poller = class {
  constructor(jobId, {interval=4000, onUpdate, onDone, onError}={}) {
    this.jobId=jobId; this.interval=interval;
    this.onUpdate=onUpdate||(() => {}); this.onDone=onDone||(() => {}); this.onError=onError||(() => {});
    this._t=null;
  }
  start() { this._t=setInterval(()=>this._poll(),this.interval); this._poll(); return this; }
  stop()  { clearInterval(this._t); }
  async _poll() {
    try {
      const d = await BJ.api.status(this.jobId);
      this.onUpdate(d);
      if(d.status==='done'){this.stop();this.onDone(d.result);}
      else if(['error','cancelled'].includes(d.status)){this.stop();this.onError(d.error||d.status);}
    } catch(e){ console.warn('[Poller]',e); }
  }
};

// Utils
BJ.utils = {
  toBase64: f => new Promise((res,rej)=>{const r=new FileReader();r.onload=e=>res(e.target.result.split(',')[1]);r.onerror=rej;r.readAsDataURL(f);}),
  sleep: ms => new Promise(r=>setTimeout(r,ms)),
  debounce: (fn,ms) => {let t;return (...a)=>{clearTimeout(t);t=setTimeout(()=>fn(...a),ms);};},
  pick: arr => arr[Math.floor(Math.random()*arr.length)],
  formatDur: s => s<60?Math.round(s)+'s':(Math.floor(s/60)+'m'+Math.round(s%60)+'s'),
  formatMB: b => (b/1048576).toFixed(1)+' MB',
};

})(window);/* BENY-JOE CINE IA PRO v5.0 - Animation & UI Library */
(function(global){'use strict';
const BJ = global.BenyJoe = global.BenyJoe || {};
BJ.VERSION = '5.0.0';
BJ.STYLES = ['cinematic','cyberpunk','epic_battle','nature','scifi','noir','fantasy','horror','romantique'];
BJ.VOICES = ['masculin','feminin','dramatique','jeune','epique'];
BJ.MUSIC  = ['cinematique','ambiante','cyberpunk','nature','scifi','noir','fantasy','horror','epic_battle','romantique'];

// Easing functions
BJ.ease = {
  linear:      t => t,
  quad:        t => t*t,
  cubic:       t => t*t*t,
  quart:       t => t*t*t*t,
  sine:        t => 1 - Math.cos(t * Math.PI / 2),
  expo:        t => t === 0 ? 0 : Math.pow(2, 10 * (t - 1)),
  circ:        t => 1 - Math.sqrt(1 - t * t),
  back:        t => t * t * (2.7 * t - 1.7),
  elastic:     t => t === 0 ? 0 : t === 1 ? 1 : -Math.pow(2, 10*(t-1)) * Math.sin((t-1.1)*5*Math.PI),
  bounce: t => {
    if(t < 1/2.75) return 7.5625*t*t;
    if(t < 2/2.75) return 7.5625*(t-=1.5/2.75)*t+0.75;
    if(t < 2.5/2.75) return 7.5625*(t-=2.25/2.75)*t+0.9375;
    return 7.5625*(t-=2.625/2.75)*t+0.984375;
  },
};

// Animate function
BJ.animate = (opts) => {
  const {duration=400, from=0, to=1, easing='linear', onUpdate, onComplete} = opts;
  const easeFn = BJ.ease[easing] || BJ.ease.linear;
  let start = null;
  const step = (timestamp) => {
    if(!start) start = timestamp;
    const progress = Math.min((timestamp - start) / duration, 1);
    const value = from + (to - from) * easeFn(progress);
    if(onUpdate) onUpdate(value, progress);
    if(progress < 1) requestAnimationFrame(step);
    else if(onComplete) onComplete();
  };
  requestAnimationFrame(step);
};

// Progress bar
BJ.ProgressBar = class {
  constructor(el) { this.el = typeof el === 'string' ? document.getElementById(el) : el; this.pct = 0; }
  set(pct, step='') {
    this.pct = Math.max(0, Math.min(100, pct));
    if(!this.el) return;
    this.el.innerHTML = '<div style="display:flex;justify-content:space-between;font-size:12px;color:#8888aa;margin-bottom:6px"><span>' + step + '</span><span>' + Math.round(this.pct) + '%</span></div><div style="background:#1a1a28;border-radius:20px;height:8px;overflow:hidden"><div style="height:100%;width:' + this.pct + '%;background:linear-gradient(90deg,#7c5cbf,#5b8dee);border-radius:20px;transition:width .4s"></div></div>';
  }
  animate(from, to, duration=400) {
    BJ.animate({ from, to, duration, easing:'quad', onUpdate: (v) => this.set(v) });
  }
};

// Toast
BJ.toast = (msg, type='ok', ms=4000) => {
  let wrap = document.getElementById('_bjt');
  if(!wrap){wrap=document.createElement('div');wrap.id='_bjt';Object.assign(wrap.style,{position:'fixed',bottom:'20px',right:'20px',zIndex:'9999',display:'flex',flexDirection:'column',gap:'8px'});document.body.appendChild(wrap);}
  const el=document.createElement('div');
  el.textContent=msg;
  Object.assign(el.style,{background:type==='err'?'#2a1020':'#102a20',border:'1px solid '+(type==='err'?'#e84b6e':'#4be8a0'),color:type==='err'?'#e84b6e':'#4be8a0',borderRadius:'10px',padding:'12px 18px',fontSize:'13px',maxWidth:'300px',boxShadow:'0 4px 20px rgba(0,0,0,.5)'});
  wrap.appendChild(el);setTimeout(()=>el.remove(),ms);
};

// API Client
BJ.api = {
  base: '',
  call: async (method, path, data) => {
    const opts = { method, headers: {'Content-Type':'application/json'} };
    if(data) opts.body = JSON.stringify(data);
    const r = await fetch(BJ.api.base + path, opts);
    return r.json();
  },
  auth:      (pin) => BJ.api.call('POST','/api/auth',{pin}),
  health:    () => BJ.api.call('GET','/api/health'),
  generate:  (p) => BJ.api.call('POST','/api/generate',p),
  image:     (p) => BJ.api.call('POST','/api/generate_image',p),
  img2video: (p) => BJ.api.call('POST','/api/img2video',p),
  status:    (id) => BJ.api.call('GET','/api/status/'+id),
  cancel:    (id) => BJ.api.call('POST','/api/cancel/'+id),
  gpuUrl:    (url) => BJ.api.call('POST','/api/gpu_url',{url}),
  history:   () => BJ.api.call('GET','/api/history'),
};

// Poller
BJ.Poller = class {
  constructor(jobId, {interval=4000, onUpdate, onDone, onError}={}) {
    this.jobId=jobId; this.interval=interval;
    this.onUpdate=onUpdate||(() => {}); this.onDone=onDone||(() => {}); this.onError=onError||(() => {});
    this._t=null;
  }
  start() { this._t=setInterval(()=>this._poll(),this.interval); this._poll(); return this; }
  stop()  { clearInterval(this._t); }
  async _poll() {
    try {
      const d = await BJ.api.status(this.jobId);
      this.onUpdate(d);
      if(d.status==='done'){this.stop();this.onDone(d.result);}
      else if(['error','cancelled'].includes(d.status)){this.stop();this.onError(d.error||d.status);}
    } catch(e){ console.warn('[Poller]',e); }
  }
};

// Utils
BJ.utils = {
  toBase64: f => new Promise((res,rej)=>{const r=new FileReader();r.onload=e=>res(e.target.result.split(',')[1]);r.onerror=rej;r.readAsDataURL(f);}),
  sleep: ms => new Promise(r=>setTimeout(r,ms)),
  debounce: (fn,ms) => {let t;return (...a)=>{clearTimeout(t);t=setTimeout(()=>fn(...a),ms);};},
  pick: arr => arr[Math.floor(Math.random()*arr.length)],
  formatDur: s => s<60?Math.round(s)+'s':(Math.floor(s/60)+'m'+Math.round(s%60)+'s'),
  formatMB: b => (b/1048576).toFixed(1)+' MB',
};

})(window);/* BENY-JOE CINE IA PRO v5.0 - Animation & UI Library */
(function(global){'use strict';
const BJ = global.BenyJoe = global.BenyJoe || {};
BJ.VERSION = '5.0.0';
BJ.STYLES = ['cinematic','cyberpunk','epic_battle','nature','scifi','noir','fantasy','horror','romantique'];
BJ.VOICES = ['masculin','feminin','dramatique','jeune','epique'];
BJ.MUSIC  = ['cinematique','ambiante','cyberpunk','nature','scifi','noir','fantasy','horror','epic_battle','romantique'];

// Easing functions
BJ.ease = {
  linear:      t => t,
  quad:        t => t*t,
  cubic:       t => t*t*t,
  quart:       t => t*t*t*t,
  sine:        t => 1 - Math.cos(t * Math.PI / 2),
  expo:        t => t === 0 ? 0 : Math.pow(2, 10 * (t - 1)),
  circ:        t => 1 - Math.sqrt(1 - t * t),
  back:        t => t * t * (2.7 * t - 1.7),
  elastic:     t => t === 0 ? 0 : t === 1 ? 1 : -Math.pow(2, 10*(t-1)) * Math.sin((t-1.1)*5*Math.PI),
  bounce: t => {
    if(t < 1/2.75) return 7.5625*t*t;
    if(t < 2/2.75) return 7.5625*(t-=1.5/2.75)*t+0.75;
    if(t < 2.5/2.75) return 7.5625*(t-=2.25/2.75)*t+0.9375;
    return 7.5625*(t-=2.625/2.75)*t+0.984375;
  },
};

// Animate function
BJ.animate = (opts) => {
  const {duration=400, from=0, to=1, easing='linear', onUpdate, onComplete} = opts;
  const easeFn = BJ.ease[easing] || BJ.ease.linear;
  let start = null;
  const step = (timestamp) => {
    if(!start) start = timestamp;
    const progress = Math.min((timestamp - start) / duration, 1);
    const value = from + (to - from) * easeFn(progress);
    if(onUpdate) onUpdate(value, progress);
    if(progress < 1) requestAnimationFrame(step);
    else if(onComplete) onComplete();
  };
  requestAnimationFrame(step);
};

// Progress bar
BJ.ProgressBar = class {
  constructor(el) { this.el = typeof el === 'string' ? document.getElementById(el) : el; this.pct = 0; }
  set(pct, step='') {
    this.pct = Math.max(0, Math.min(100, pct));
    if(!this.el) return;
    this.el.innerHTML = '<div style="display:flex;justify-content:space-between;font-size:12px;color:#8888aa;margin-bottom:6px"><span>' + step + '</span><span>' + Math.round(this.pct) + '%</span></div><div style="background:#1a1a28;border-radius:20px;height:8px;overflow:hidden"><div style="height:100%;width:' + this.pct + '%;background:linear-gradient(90deg,#7c5cbf,#5b8dee);border-radius:20px;transition:width .4s"></div></div>';
  }
  animate(from, to, duration=400) {
    BJ.animate({ from, to, duration, easing:'quad', onUpdate: (v) => this.set(v) });
  }
};

// Toast
BJ.toast = (msg, type='ok', ms=4000) => {
  let wrap = document.getElementById('_bjt');
  if(!wrap){wrap=document.createElement('div');wrap.id='_bjt';Object.assign(wrap.style,{position:'fixed',bottom:'20px',right:'20px',zIndex:'9999',display:'flex',flexDirection:'column',gap:'8px'});document.body.appendChild(wrap);}
  const el=document.createElement('div');
  el.textContent=msg;
  Object.assign(el.style,{background:type==='err'?'#2a1020':'#102a20',border:'1px solid '+(type==='err'?'#e84b6e':'#4be8a0'),color:type==='err'?'#e84b6e':'#4be8a0',borderRadius:'10px',padding:'12px 18px',fontSize:'13px',maxWidth:'300px',boxShadow:'0 4px 20px rgba(0,0,0,.5)'});
  wrap.appendChild(el);setTimeout(()=>el.remove(),ms);
};

// API Client
BJ.api = {
  base: '',
  call: async (method, path, data) => {
    const opts = { method, headers: {'Content-Type':'application/json'} };
    if(data) opts.body = JSON.stringify(data);
    const r = await fetch(BJ.api.base + path, opts);
    return r.json();
  },
  auth:      (pin) => BJ.api.call('POST','/api/auth',{pin}),
  health:    () => BJ.api.call('GET','/api/health'),
  generate:  (p) => BJ.api.call('POST','/api/generate',p),
  image:     (p) => BJ.api.call('POST','/api/generate_image',p),
  img2video: (p) => BJ.api.call('POST','/api/img2video',p),
  status:    (id) => BJ.api.call('GET','/api/status/'+id),
  cancel:    (id) => BJ.api.call('POST','/api/cancel/'+id),
  gpuUrl:    (url) => BJ.api.call('POST','/api/gpu_url',{url}),
  history:   () => BJ.api.call('GET','/api/history'),
};

// Poller
BJ.Poller = class {
  constructor(jobId, {interval=4000, onUpdate, onDone, onError}={}) {
    this.jobId=jobId; this.interval=interval;
    this.onUpdate=onUpdate||(() => {}); this.onDone=onDone||(() => {}); this.onError=onError||(() => {});
    this._t=null;
  }
  start() { this._t=setInterval(()=>this._poll(),this.interval); this._poll(); return this; }
  stop()  { clearInterval(this._t); }
  async _poll() {
    try {
      const d = await BJ.api.status(this.jobId);
      this.onUpdate(d);
      if(d.status==='done'){this.stop();this.onDone(d.result);}
      else if(['error','cancelled'].includes(d.status)){this.stop();this.onError(d.error||d.status);}
    } catch(e){ console.warn('[Poller]',e); }
  }
};

// Utils
BJ.utils = {
  toBase64: f => new Promise((res,rej)=>{const r=new FileReader();r.onload=e=>res(e.target.result.split(',')[1]);r.onerror=rej;r.readAsDataURL(f);}),
  sleep: ms => new Promise(r=>setTimeout(r,ms)),
  debounce: (fn,ms) => {let t;return (...a)=>{clearTimeout(t);t=setTimeout(()=>fn(...a),ms);};},
  pick: arr => arr[Math.floor(Math.random()*arr.length)],
  formatDur: s => s<60?Math.round(s)+'s':(Math.floor(s/60)+'m'+Math.round(s%60)+'s'),
  formatMB: b => (b/1048576).toFixed(1)+' MB',
};

})(window);
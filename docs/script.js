/* ================= INTERACTIVE FIGURES ================= */
(function(){
"use strict";
var REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
var SVGNS = 'http://www.w3.org/2000/svg';
function h(tag, cls, html){ var e=document.createElement(tag); if(cls)e.className=cls; if(html!=null)e.innerHTML=html; return e; }
function s(tag, attrs, parent){ var e=document.createElementNS(SVGNS,tag); for(var k in attrs)e.setAttribute(k,attrs[k]); if(parent)parent.appendChild(e); return e; }
function txt(parent, x, y, str, size, fill, anchor, weight, mono){
  var t=s('text',{x:x,y:y,'font-size':size||12,fill:fill||'#38445F','text-anchor':anchor||'start',
    'font-family':mono?'"IBM Plex Mono",monospace':'"IBM Plex Sans",sans-serif','font-weight':weight||400},parent);
  t.textContent=str; return t;
}

/* ---------- reveal-on-scroll: each viz registers a callback ---------- */
var reveals = {};
var io = ('IntersectionObserver' in window) ? new IntersectionObserver(function(entries){
  entries.forEach(function(en){
    if(en.isIntersecting){
      en.target.classList.add('in');
      var fn = reveals[en.target.id]; if(fn){ fn(); delete reveals[en.target.id]; }
      io.unobserve(en.target);
    }
  });
},{threshold:.25}) : null;
function onReveal(el, fn){
  if(!io){ el.classList.add('in'); fn(); return; }
  reveals[el.id]=fn; io.observe(el);
}

/* =====================================================================
   1 · HOME vs AWAY dumbbells  (domain 40..100)
===================================================================== */
(function(){
  var box=document.getElementById('viz-homeaway'); if(!box) return;
  var wrap=box.querySelector('.dumb');
  var X=function(v){ return (v-40)/60*100; };
  var rows=[
    {n:'CICIDS', s:'large corporate network', away:45.3, home:94.6},
    {n:'UNSW',   s:'corporate network',       away:59.6, home:88.7},
    {n:'BoT-IoT',s:'smart devices',           away:56.8, home:72.2},
    {n:'ToN-IoT',s:'smart devices + office',  away:68.8, home:77.6}
  ];
  var anims=[];
  rows.forEach(function(r){
    var row=h('div','drow'), lab=h('div','dlab',r.n+'<small>'+r.s+'</small>'), tr=h('div','dtrack');
    tr.appendChild(h('span','flip')).style.left=X(50)+'%';
    var line=h('span','dline'); line.style.left=X(r.away)+'%'; tr.appendChild(line);
    var pa=h('span','pt away'); pa.style.left=X(r.away)+'%'; pa.title=r.n+' on the other three networks: '+r.away; tr.appendChild(pa);
    var ph=h('span','pt home'); ph.style.left=X(r.away)+'%'; ph.title=r.n+' on its own network: '+r.home; tr.appendChild(ph);
    var va=h('span','dval dn',r.away.toFixed(1)); va.style.left=X(r.away)+'%'; tr.appendChild(va);
    var vh=h('span','dval up',r.home.toFixed(1)); vh.style.left=X(r.away)+'%'; vh.style.color='#0f7d96'; vh.style.fontWeight='600'; tr.appendChild(vh);
    row.appendChild(lab); row.appendChild(tr); wrap.appendChild(row);
    anims.push(function(){ line.style.width=(X(r.home)-X(r.away))+'%'; ph.style.left=X(r.home)+'%'; vh.style.left=X(r.home)+'%'; });
  });
  onReveal(box,function(){ anims.forEach(function(f){f();}); });
})();

/* =====================================================================
   2 · ATTACKS CAUGHT vs FALSE ALARMS paired bars
===================================================================== */
(function(){
  var box=document.getElementById('viz-caught'); if(!box) return;
  var wrap=box.querySelector('.pbars');
  var rows=[
    {n:'CICIDS', s:'large corporate network', caught:87.8, alarms:9.9},
    {n:'UNSW',   s:'corporate network',       caught:70.1, alarms:5.4, hot:true},
    {n:'BoT-IoT',s:'smart devices',           caught:95.8, alarms:41.4, hot:true},
    {n:'ToN-IoT',s:'smart devices + office',  caught:70.4, alarms:16.9}
  ];
  var anims=[];
  rows.forEach(function(r){
    var row=h('div','pb-row'), lab=h('div','dlab',r.n+'<small>'+r.s+'</small>'), bars=h('div');
    [['catch',r.caught,'% of attacks caught'],['alarm',r.alarms,'% false alarms']].forEach(function(cfg){
      var pb=h('div','pb'), fill=h('i','fill '+cfg[0]), b=h('b');
      b.textContent=cfg[1].toFixed(0)+cfg[2];
      var inside = cfg[1]>62;
      b.style.left='8px'; b.style.color=inside?'#fff':'#38445F';
      pb.appendChild(fill); pb.appendChild(b); bars.appendChild(pb);
      anims.push(function(){
        fill.style.width=cfg[1]+'%';
        b.style.left = inside ? 'calc('+cfg[1]+'% - 8px)' : 'calc('+cfg[1]+'% + 9px)';
        if(inside) b.style.transform='translate(-100%,-50%)';
      });
    });
    if(r.hot){ row.style.background='linear-gradient(90deg,rgba(255,107,74,.06),transparent)'; row.style.borderRadius='10px'; }
    row.appendChild(lab); row.appendChild(bars); wrap.appendChild(row);
  });
  onReveal(box,function(){ anims.forEach(function(f){f();}); });
})();

/* =====================================================================
   3+4+5 · SIMPLE BAR CHARTS (insights 03, 04, 05)
===================================================================== */
function simpleBars(boxId, rows, cfg){
  var box=document.getElementById(boxId); if(!box) return;
  var wrap=box.querySelector('.rank');
  var X=function(v){ return (v-cfg.min)/(cfg.max-cfg.min)*100; };
  var anims=[];
  rows.forEach(function(r,i){
    var row=h('div','rrow'+(r.cls?' '+r.cls:''));
    row.appendChild(h('div','rlab',r.n+(r.s?'<small>'+r.s+'</small>':'')));
    var tr=h('div','rtrack');
    var bar=h('span','rbar'); bar.title=r.n+': '+r.v.toFixed(1); tr.appendChild(bar);
    var val=h('span','rval',r.v.toFixed(1)); val.style.left='10px'; tr.appendChild(val);
    if(cfg.inside){ val.style.color = r.cls==='lead' ? '#fff' : '#38445F'; val.style.fontWeight='600'; }
    (cfg.refs||[]).forEach(function(rf){
      var ref=h('span','rref'); ref.style.left=X(rf.v)+'%';
      var labelRow = rf.on==='last' ? rows.length-1 : 0;
      if(i===labelRow) ref.appendChild(h('span', rf.on==='last'?'below':'', rf.label+' · '+rf.v.toFixed(1)));
      tr.appendChild(ref);
    });
    bar.style.transitionDelay=(i*.25)+'s'; val.style.transitionDelay=(i*.25)+'s';
    row.appendChild(tr); wrap.appendChild(row);
    anims.push(function(){
      bar.style.width=X(r.v)+'%';
      if(cfg.inside){ val.style.left='calc('+X(r.v)+'% - 9px)'; val.style.transform='translate(-100%,-50%)'; }
      else val.style.left='calc('+Math.min(X(r.v),86)+'% + 10px)';
    });
  });
  onReveal(box,function(){ anims.forEach(function(f){f();}); });
}

simpleBars('viz-pool',[
  {n:'No sharing', s:'each network trains its own model', v:83.3, cls:'base'},
  {n:'All data pooled', s:'one model, everything mixed together', v:73.4, cls:'worst'}
],{min:50,max:90});

simpleBars('viz-personal',[
  {n:'Pooled + a personal part', s:'shared knowledge, personal rebuilding', v:84.6, cls:'lead'},
  {n:'No sharing', s:'each network trains its own model', v:83.3, cls:'base'},
  {n:'All data pooled', s:'one model, everything mixed together', v:73.4, cls:'worst'}
],{min:50,max:90});

simpleBars('viz-fedbest',[
  {n:'Share all except scaling', s:'FedBN — the animation’s recipe', v:85.2, cls:'lead'},
  {n:'Share encoder, keep decoder local', s:'FedRep', v:83.3},
  {n:'Share the whole model', s:'nothing kept personal', v:81.8}
],{min:50,max:90, inside:true, refs:[
  {v:84.6,label:'best pooled'},
  {v:83.3,label:'no sharing', on:'last'}
]});

/* =====================================================================
   5 · TRANSFER arrows (1 teacher → 3 teachers, ceiling diamond)
===================================================================== */
(function(){
  var box=document.getElementById('viz-transfer'); if(!box) return;
  var wrap=box.querySelector('.dumb');
  var X=function(v){ return (v-40)/60*100; };
  var rows=[
    {n:'All four (average)', s:'', one:57.6, three:75.2, bench:83.3, avg:true},
    {n:'CICIDS',  s:'large corporate network', one:45.3, three:76.9, bench:94.6},
    {n:'UNSW',    s:'corporate network',       one:59.6, three:81.9, bench:88.7},
    {n:'BoT-IoT', s:'smart devices',           one:56.8, three:65.3, bench:72.2},
    {n:'ToN-IoT', s:'smart devices + office',  one:68.8, three:76.7, bench:77.6}
  ];
  var anims=[];
  rows.forEach(function(r){
    var row=h('div','drow'+(r.avg?' avg':'')), lab=h('div','dlab',r.n+(r.s?'<small>'+r.s+'</small>':'')), tr=h('div','dtrack');
    tr.appendChild(h('span','flip')).style.left=X(50)+'%';
    var line=h('span','dline arr'); line.style.left=X(r.one)+'%'; tr.appendChild(line);
    var bench=h('span','bench'); bench.style.left=X(r.bench)+'%'; bench.title='Ceiling (trained on the network itself): '+r.bench; tr.appendChild(bench);
    var p1=h('span','pt away'); p1.style.left=X(r.one)+'%'; p1.title='Trained on 1 other network: '+r.one; tr.appendChild(p1);
    var p3=h('span','pt newm'); p3.style.left=X(r.one)+'%'; p3.title='Trained on the other 3 networks: '+r.three; tr.appendChild(p3);
    var d=h('span','dval up delta','+'+(r.three-r.one).toFixed(1)); d.style.left=X(r.one)+'%'; tr.appendChild(d);
    var v1=h('span','dval dn',r.one.toFixed(1)); v1.style.left=X(r.one)+'%'; tr.appendChild(v1);
    row.appendChild(lab); row.appendChild(tr); wrap.appendChild(row);
    anims.push(function(){ line.style.width=(X(r.three)-X(r.one))+'%'; p3.style.left=X(r.three)+'%'; d.style.left=X(r.three)+'%'; });
  });
  onReveal(box,function(){ anims.forEach(function(f){f();}); });
})();

/* =====================================================================
   6 · SecAgg mini deltas
===================================================================== */
(function(){
  var box=document.getElementById('viz-secagg'); if(!box) return;
  var rows=[ ['BoT-IoT',3.3],['UNSW',2.0],['ToN-IoT',0.5],['CICIDS',0.2] ];
  var anims=[];
  rows.forEach(function(r){
    var row=h('div','md-row');
    row.appendChild(h('span','',r[0]));
    var trk=h('div','md-track'), fill=h('i','md-fill'); trk.appendChild(fill); row.appendChild(trk);
    row.appendChild(h('span','md-val','+'+r[1].toFixed(1)));
    box.appendChild(row);
    anims.push(function(){ fill.style.width=(r[1]/20*100)+'%'; });
  });
  onReveal(box,function(){ anims.forEach(function(f){f();}); });
})();

/* =====================================================================
   7 · DP privacy-dial line chart (SVG)
===================================================================== */
(function(){
  var box=document.getElementById('viz-dp'); if(!box) return;
  var wrap=box.querySelector('.dp-chart');
  var sig=[0.1,0.15,0.2,0.25,0.3,0.5,0.7,1,3,10];
  var score=[83.6,78.2,66.1,70.4,62.8,61.1,63.8,53.9,55.9,53.5];
  var eps=[2800,1300,790,540,390,160,92,55,12,3.5];
  var W=760,H=310,L=46,R=16,T=26,B=64, BASE=83.8;
  var x=function(i){ return L+i/(sig.length-1)*(W-L-R); };
  var y=function(v){ return T+(90-v)/(90-45)*(H-T-B); };
  var svg=s('svg',{viewBox:'0 0 '+W+' '+H,role:'img','aria-label':'Line chart: ROC-AUC falls from 84 to about 53 as differential-privacy noise increases'});
  wrap.appendChild(svg);
  // meaningful-privacy zone (last two settings)
  var zx=x(7.5);
  s('rect',{x:zx,y:T-6,width:W-R-zx,height:H-T-B+6,fill:'rgba(124,107,255,.07)',rx:8},svg);
  txt(svg,(zx+W-R)/2,T+8,'privacy actually',10.5,'#7C6BFF','middle',600);
  txt(svg,(zx+W-R)/2,T+21,'meaningful (ε ≤ 12)',10.5,'#7C6BFF','middle',600);
  // gridlines + y labels
  [50,60,70,80].forEach(function(g){
    s('line',{x1:L,x2:W-R,y1:y(g),y2:y(g),stroke:'#E4E9F3','stroke-width':1},svg);
    txt(svg,L-8,y(g)+4,String(g),11,'#697691','end',400,true);
  });
  // coin-flip + baseline
  s('line',{x1:L,x2:W-R,y1:y(50),y2:y(50),stroke:'#E8462A','stroke-width':1.4,'stroke-dasharray':'6 5',opacity:.65},svg);
  txt(svg,W-R,y(50)-6,'coin flip (50)',10.5,'#E8462A','end',500,true);
  s('line',{x1:L,x2:W-R,y1:y(BASE),y2:y(BASE),stroke:'#38445F','stroke-width':1.3,'stroke-dasharray':'2 5'},svg);
  txt(svg,L+4,y(BASE)-7,'no noise: ROC-AUC 83.8',10.5,'#38445F','start',500,true);
  // line + dots
  var pts=score.map(function(v,i){ return x(i)+','+y(v); }).join(' ');
  var line=s('polyline',{points:pts,fill:'none',stroke:'#1BA5C4','stroke-width':2.6,'stroke-linejoin':'round','stroke-linecap':'round'},svg);
  score.forEach(function(v,i){
    var strong=i>=8;
    var c=s('circle',{cx:x(i),cy:y(v),r:5.5,fill:strong?'#7C6BFF':'#1BA5C4',stroke:'#fff','stroke-width':2},svg);
    var t=s('title',{},c);
    t.textContent='Noise level σ='+sig[i]+' → ROC-AUC '+v.toFixed(1)+' (privacy budget ε≈'+eps[i]+')';
  });
  // x axis words
  txt(svg,L,H-30,'barely any noise',11,'#697691','start',400,true);
  txt(svg,W-R,H-30,'heavy noise',11,'#697691','end',400,true);
  txt(svg,(L+W-R)/2,H-10,'→ turning the privacy dial up (more noise added to the shared model)',11.5,'#38445F','middle',500);
  // draw-in animation
  if(!REDUCED){
    var len=line.getTotalLength();
    line.style.strokeDasharray=len; line.style.strokeDashoffset=len;
    line.style.transition='stroke-dashoffset 1.8s ease';
  }
  onReveal(box,function(){ line.style.strokeDashoffset=0; });
})();

/* =====================================================================
   7b · AUTOENCODER intro demo (encoder → tiny summary → decoder → compare)
===================================================================== */
(function(){
  var svg=document.getElementById('ae-svg'); if(!svg) return;
  var cap=document.getElementById('ae-cap');
  var INK='#0F1830', MUT='#697691', SIG='#1BA5C4', VIO='#7C6BFF', BAD='#E8462A', OK='#0a8a52';
  var FIELDS=['duration','packets','bytes','TTL','port'];
  var NORM_IN =[.55,.40,.62,.48,.35], NORM_OUT=[.52,.43,.60,.50,.37];
  var ATK_IN  =[.97,.93,.99,.15,.88], ATK_OUT =[.50,.42,.57,.52,.40];
  var BW=120, ROWY=function(i){ return 96+i*30; };

  function scaled(el){ el.style.transformBox='fill-box'; el.style.transformOrigin='left center'; el.style.transform='scale(0,1)'; return el; }
  function setScale(el,v,dur){ el.style.transition='transform '+(dur||.8)+'s cubic-bezier(.22,.8,.24,1)'; el.style.transform='scale('+v+',1)'; }
  function hide(el){ el.style.transition='none'; el.style.transform='scale(0,1)'; void el.getBoundingClientRect(); }

  /* input column: the real numbers */
  txt(svg,186,62,'the connection’s real numbers',12,INK,'end',600);
  var inBars=FIELDS.map(function(f,i){
    txt(svg,118,ROWY(i)+10,f,10.5,MUT,'end',500,true);
    s('rect',{x:126,y:ROWY(i),width:BW,height:12,rx:4,fill:'#EDF1F8'},svg);
    return scaled(s('rect',{x:126,y:ROWY(i),width:0,height:12,rx:4,fill:SIG},svg));
  });

  /* encoder funnel */
  var enc=s('polygon',{points:'262,80 390,122 390,194 262,236',fill:'rgba(27,165,196,.10)',stroke:SIG,'stroke-width':1.6},svg);
  txt(svg,326,152,'ENCODER',12.5,'#0f7d96','middle',700,true);
  txt(svg,326,170,'squeeze',10.5,MUT,'middle',400,true);

  /* bottleneck: the tiny summary */
  var mids=[122,152,182].map(function(y){ return s('rect',{x:404,y:y,width:26,height:22,rx:6,fill:VIO,opacity:0},svg); });
  txt(svg,417,232,'tiny',10.5,VIO,'middle',600,true);
  txt(svg,417,246,'summary',10.5,VIO,'middle',600,true);

  /* decoder funnel */
  var dec=s('polygon',{points:'444,122 572,80 572,236 444,194',fill:'rgba(124,107,255,.10)',stroke:VIO,'stroke-width':1.6},svg);
  txt(svg,508,152,'DECODER',12.5,VIO,'middle',700,true);
  txt(svg,508,170,'rebuild',10.5,MUT,'middle',400,true);

  /* output column: the rebuild, with tick marks for the real values */
  txt(svg,588,62,'the rebuild',12,INK,'start',600);
  var tickLab=txt(svg,678,62,'| = the real value',10.5,MUT,'start',500,true);
  tickLab.setAttribute('opacity',0);
  var outBars=[], ticks=[];
  FIELDS.forEach(function(f,i){
    s('rect',{x:588,y:ROWY(i),width:BW,height:12,rx:4,fill:'#EDF1F8'},svg);
    outBars.push(scaled(s('rect',{x:588,y:ROWY(i),width:0,height:12,rx:4,fill:VIO},svg)));
    ticks.push(s('line',{x1:588,y1:ROWY(i)-4,x2:588,y2:ROWY(i)+16,stroke:INK,'stroke-width':2.4,opacity:0},svg));
  });

  /* error meter + verdict */
  txt(svg,823,100,'reconstruction error',11.5,INK,'middle',600);
  s('rect',{x:748,y:112,width:150,height:14,rx:7,fill:'#EDF1F8'},svg);
  var errFill=scaled(s('rect',{x:748,y:112,width:150,height:14,rx:7,fill:OK},svg));
  var verdict1=txt(svg,823,160,'',13.5,OK,'middle',700);
  var verdict2=txt(svg,823,180,'',10.5,MUT,'middle',400,true);

  var token=0, T=REDUCED?0:1;
  function run(mode){
    token++; var my=token, ok=mode==='n';
    var vin=ok?NORM_IN:ATK_IN, vout=ok?NORM_OUT:ATK_OUT;
    inBars.forEach(function(b,i){ hide(b); b.setAttribute('width',vin[i]*BW); });
    outBars.forEach(function(b,i){ hide(b); b.setAttribute('width',vout[i]*BW); });
    hide(errFill);
    mids.forEach(function(m){ m.style.transition='none'; m.setAttribute('opacity',0); });
    ticks.forEach(function(tk){ tk.setAttribute('opacity',0); });
    tickLab.setAttribute('opacity',0);
    enc.setAttribute('fill','rgba(27,165,196,.10)'); dec.setAttribute('fill','rgba(124,107,255,.10)');
    verdict1.textContent=''; verdict2.textContent='';
    function at(ms,fn){ setTimeout(function(){ if(my===token) fn(); }, ms*T); }
    at(60,function(){
      cap.textContent='1 · The connection’s numbers go in.';
      inBars.forEach(function(b){ setScale(b,1,.7); });
    });
    at(1100,function(){
      cap.textContent='2 · The encoder squeezes them into a tiny summary.';
      enc.setAttribute('fill','rgba(27,165,196,.28)');
      mids.forEach(function(m,i){ m.style.transition='opacity .4s ease '+(i*.12)+'s'; m.setAttribute('opacity',1); });
    });
    at(2500,function(){
      cap.textContent='3 · The decoder rebuilds the numbers from the summary alone.';
      enc.setAttribute('fill','rgba(27,165,196,.10)');
      dec.setAttribute('fill','rgba(124,107,255,.28)');
      outBars.forEach(function(b){ setScale(b,1,.8); });
    });
    at(4000,function(){
      cap.textContent='4 · Compare rebuild vs real: '+(ok?'a close match.':'a wild miss.');
      dec.setAttribute('fill','rgba(124,107,255,.10)');
      ticks.forEach(function(tk,i){ var x=588+vin[i]*BW; tk.setAttribute('x1',x); tk.setAttribute('x2',x); tk.setAttribute('opacity',1); });
      tickLab.setAttribute('opacity',1);
      errFill.setAttribute('fill', ok?OK:BAD);
      setScale(errFill, ok?.08:.94, 1.1);
      verdict1.setAttribute('fill', ok?OK:BAD);
      verdict1.textContent= ok?'✓ tiny error — looks normal':'✗ huge error — flagged';
      verdict2.textContent= ok?'rebuild ≈ the real values':'it can only rebuild “normal”';
    });
  }
  var bn=document.getElementById('ae-normal'), ba=document.getElementById('ae-attack');
  function sel(mode){
    bn.classList.toggle('on',mode==='n'); ba.classList.toggle('on',mode!=='n');
    run(mode);
  }
  bn.addEventListener('click',function(){ sel('n'); });
  ba.addEventListener('click',function(){ sel('a'); });
  onReveal(document.getElementById('ae-demo'),function(){ run('n'); });
})();

/* =====================================================================
   8 · PIPELINE interactive stage
===================================================================== */
(function(){
  var svg=document.getElementById('pipe-svg'); if(!svg) return;
  var capN=document.getElementById('pipe-cap-n'), capT=document.getElementById('pipe-cap-t'), cap=document.getElementById('pipe-cap');
  var roleEl=document.getElementById('pipe-role');
  var INK='#0F1830', MUT='#697691', SIG='#1BA5C4', VIO='#7C6BFF', ANOM='#FF6B4A', DIM='#C7D2E4';

  function g(id){ var e=s('g',{id:id,'class':'pel'},svg); return e; }
  var defs=s('defs',{},svg);
  function marker(id,color){
    var mk=s('marker',{id:id,markerWidth:10,markerHeight:9,refX:9,refY:4.5,orient:'auto',markerUnits:'userSpaceOnUse'},defs);
    s('path',{d:'M0,0 L10,4.5 L0,9 z',fill:color},mk);
  }
  marker('pArr',MUT); marker('mGrey','#9DB1CD'); marker('mCyan',SIG); marker('mRed',ANOM); marker('mDim',DIM);

  /* -- records table -- */
  var gT=g('p-table');
  s('rect',{x:22,y:64,width:224,height:190,rx:12,fill:'#F5F7FB',stroke:'#DCE3EF'},gT);
  txt(gT,134,50,'1,000 connection records',13,INK,'middle',600);
  for(var i=0;i<7;i++){
    var ry=78+i*25;
    s('rect',{x:34,y:ry,width:200,height:18,rx:4,fill:i===0?'#DCE3EF':'#fff',stroke:'#E4E9F3'},gT);
    if(i>0){
      s('rect',{x:40,y:ry+5,width:44,height:8,rx:3,fill:'#B9C6DE'},gT);   /* from */
      s('rect',{x:92,y:ry+5,width:44,height:8,rx:3,fill:'#B9C6DE'},gT);   /* to */
      s('rect',{x:144,y:ry+5,width:24+((i*37)%42),height:8,rx:3,fill:'#8FD8E8'},gT); /* size */
    } else {
      txt(gT,40,ry+13,'from',9,MUT,'start',500,true); txt(gT,92,ry+13,'to',9,MUT,'start',500,true);
      txt(gT,144,ry+13,'bytes, ports…',9,MUT,'start',500,true);
    }
  }
  var gA=g('p-arrow');
  s('path',{d:'M254 160 H300',stroke:MUT,'stroke-width':2,fill:'none','marker-end':'url(#pArr)'},gA);

  /* -- directed graph: dots = devices, arrows = connections -- */
  var N=[[398,96],[520,70],[468,178],[372,256],[532,262],[608,160]];
  var E=[[0,1],[0,2],[1,5],[2,3],[2,4],[3,4],[4,5],[2,5],[1,2]];
  var BAD=2;   /* edge [1,5]: the anomaly flagged at step 5 */
  var GUESS=4; /* edge [2,4]: the example the model guesses at step 4 */
  var gG=g('p-graph'), edgeEls=[];
  E.forEach(function(e){
    var a=N[e[0]], b=N[e[1]], dx=b[0]-a[0], dy=b[1]-a[1], L=Math.sqrt(dx*dx+dy*dy), ux=dx/L, uy=dy/L;
    edgeEls.push(s('line',{x1:a[0]+ux*12,y1:a[1]+uy*12,x2:b[0]-ux*16,y2:b[1]-uy*16,
      stroke:'#9DB1CD','stroke-width':2,'marker-end':'url(#mGrey)','class':'pipe-edge'},gG));
  });
  var gH=g('p-halos');
  N.forEach(function(p){
    s('circle',{cx:p[0],cy:p[1],r:19,fill:'none',stroke:VIO,'stroke-width':3,opacity:.55,'class':'pulse-halo'},gH);
    /* tiny bar-chart chip = that device's behaviour profile */
    var bx=p[0]+14, by=p[1]-36;
    s('rect',{x:bx,y:by,width:27,height:22,rx:5,fill:'#fff',stroke:VIO,'stroke-width':1.2},gH);
    for(var i=0;i<3;i++){
      var hgt=5+((p[0]*(i+3)+p[1]*(i+1))%10);
      s('rect',{x:bx+4+i*7,y:by+18-hgt,width:5,height:hgt,rx:1,fill:VIO,opacity:.8},gH);
    }
  });
  txt(gH,372,330,'every device gets a behaviour profile: a small summary of how it usually acts',12.5,VIO,'start',600);
  N.forEach(function(p){ s('circle',{cx:p[0],cy:p[1],r:8,fill:'#fff',stroke:SIG,'stroke-width':3},gG); });
  var gGlab=txt(gG,372,306,'dots = devices · arrows = connections, sender → receiver',12.5,MUT,'start',500);

  /* -- network summary -- */
  var gC=g('p-ctx');
  E.slice(0,6).forEach(function(e){
    var mx=(N[e[0]][0]+N[e[1]][0])/2, my=(N[e[0]][1]+N[e[1]][1])/2;
    s('path',{d:'M'+mx+' '+my+' Q 730 40 762 74',stroke:VIO,'stroke-width':1.3,'stroke-dasharray':'3 5',fill:'none',opacity:.6},gC);
  });
  s('rect',{x:742,y:64,width:152,height:52,rx:12,fill:'rgba(124,107,255,.1)',stroke:VIO,'stroke-width':1.5},gC);
  txt(gC,818,86,'network summary',13,VIO,'middle',600);
  txt(gC,818,104,'all connections, averaged',10,MUT,'middle',400,true);

  /* -- guess-vs-real panel, shared by steps 4 and 5 -- */
  function panel(gp,title,rows,verdict,vcol,note){
    s('rect',{x:618,y:136,width:244,height:190,rx:12,fill:'#F5F7FB',stroke:'#DCE3EF'},gp);
    txt(gp,740,159,title,12.5,INK,'middle',600);
    txt(gp,772,181,'rebuilt',10,VIO,'end',600,true);
    txt(gp,848,181,'real',10,MUT,'end',600,true);
    rows.forEach(function(r,i){
      var y=204+i*24;
      txt(gp,632,y,r[0],10.5,MUT,'start',500,true);
      txt(gp,772,y,r[1],11.5,VIO,'end',600,true);
      txt(gp,848,y,r[2],11.5,'#38445F','end',600,true);
    });
    txt(gp,632,286,verdict,12.5,vcol,'start',700);
    if(note) txt(gp,632,306,note,9.5,MUT,'start',400,true);
  }

  /* -- step 4: guess one connection's numbers -- */
  var gD=g('p-guess');
  var q0=N[E[GUESS][0]], q1=N[E[GUESS][1]], qx=(q0[0]+q1[0])/2, qy=(q0[1]+q1[1])/2;
  s('circle',{cx:q0[0],cy:q0[1],r:13,fill:'none',stroke:VIO,'stroke-width':2.5},gD);
  s('circle',{cx:q1[0],cy:q1[1],r:13,fill:'none',stroke:VIO,'stroke-width':2.5},gD);
  s('path',{d:'M'+(qx+14)+' '+qy+' Q 585 231 614 231',stroke:MUT,'stroke-width':1.4,'stroke-dasharray':'4 4',fill:'none'},gD);
  panel(gD,'rebuild this arrow’s numbers',
    [['duration','0.41 s','0.42 s'],['packets','11','12'],['bytes','980 B','1,004 B']],
    '✓ close, looks normal','#0a8a52','using only the profiles + summary');

  /* -- step 5: compare & flag -- */
  var gS=g('p-score');
  var b0=N[E[BAD][0]], b1=N[E[BAD][1]], bx=(b0[0]+b1[0])/2, by=(b0[1]+b1[1])/2;
  s('circle',{cx:bx,cy:by,r:15,fill:ANOM},gS);
  txt(gS,bx,by+5,'!',15,'#fff','middle',700);
  s('path',{d:'M'+(bx+18)+' '+(by+8)+' Q 600 138 616 165',stroke:'#E8462A','stroke-width':1.4,'stroke-dasharray':'4 4',fill:'none'},gS);
  panel(gS,'the flagged arrow’s numbers',
    [['duration','0.4 s','47 s'],['packets','12','9,214'],['bytes','1 kB','9.6 MB']],
    '✗ wild miss → flagged','#E8462A','nothing it learned looks like this');
  txt(gS,22,306,'close rebuild = normal',12.5,MUT,'start',500);
  txt(gS,22,326,'wild miss = flagged',12.5,'#E8462A','start',600);

  var CAPS=[
    ['Draw the graph.','Take 1,000 connection records and turn them into a directed graph: each device a dot, each connection an arrow from sender to receiver.','prep'],
    ['Profile each device.','The encoder squeezes the connections around each device into a behaviour profile: a tiny summary of how that device usually acts.','enc'],
    ['Summarise the moment.','The encoder also squeezes all 1,000 connections into one network summary: the gist of what’s going on right now.','enc'],
    ['Rebuild every connection.','The decoder hides each connection’s real numbers and rebuilds them (duration, packets, bytes) from just the two profiles and the summary.','dec'],
    ['Compare and flag.','Rebuild vs reality: on normal traffic the rebuild lands close. A wild miss means the model has never seen anything like it, so it gets flagged.','score']
  ];
  var ROLES={
    prep:['SET-UP','#697691','rgba(105,118,145,.12)'],
    enc:['ENCODER · squeeze','#0f7d96','rgba(27,165,196,.13)'],
    dec:['DECODER · rebuild','#7C6BFF','rgba(124,107,255,.13)'],
    score:['THE VERDICT','#E8462A','rgba(255,107,74,.13)']
  };
  var SHOW={
    1:{on:['p-table','p-arrow','p-graph'],dim:[]},
    2:{on:['p-graph','p-halos'],dim:['p-table','p-arrow']},
    3:{on:['p-graph','p-ctx'],dim:['p-table','p-arrow','p-halos']},
    4:{on:['p-graph','p-ctx','p-guess'],dim:['p-table','p-arrow']},
    5:{on:['p-graph','p-score'],dim:['p-table','p-arrow','p-ctx']}
  };
  var dotsBox=document.getElementById('pipe-dots');
  for(var di=1;di<=5;di++){
    var db=document.createElement('button');
    db.dataset.step=di; db.setAttribute('aria-label','Step '+di);
    dotsBox.appendChild(db);
  }
  var current=1;
  function setStep(n){
    current=n;
    ['p-table','p-arrow','p-graph','p-halos','p-ctx','p-guess','p-score'].forEach(function(id){
      var e=document.getElementById(id);
      e.classList.remove('on'); e.style.opacity='';
      if(SHOW[n].on.indexOf(id)>=0) e.classList.add('on');
      else if(SHOW[n].dim.indexOf(id)>=0){ e.classList.add('on'); e.style.opacity=.18; }
    });
    edgeEls.forEach(function(el,i){
      var stroke='#9DB1CD', mk='mGrey', w=2, op=1;
      if(n===5){ if(i===BAD){stroke=ANOM;mk='mRed';w=4;op=1;} else {stroke=SIG;mk='mCyan';w=2.2;op=.8;} }
      else if(n===4){ if(i===GUESS){stroke=SIG;mk='mCyan';w=3;op=1;} else {stroke=DIM;mk='mDim';w=2;op=.6;} }
      el.setAttribute('stroke',stroke); el.setAttribute('marker-end','url(#'+mk+')');
      el.setAttribute('stroke-width',w); el.setAttribute('opacity',op);
    });
    gGlab.setAttribute('opacity', n>=4?0:1);
    capN.textContent='STEP '+n+' / 5'; capT.textContent=CAPS[n-1][0]; cap.textContent=CAPS[n-1][1];
    var role=ROLES[CAPS[n-1][2]];
    roleEl.textContent=role[0]; roleEl.style.color=role[1]; roleEl.style.background=role[2];
    dotsBox.querySelectorAll('button').forEach(function(b){ b.classList.toggle('on',+b.dataset.step===n); });
  }
  var auto=null;
  function stopAuto(){ if(auto){ clearInterval(auto); auto=null; } }
  document.getElementById('pipe-prev').addEventListener('click',function(){ stopAuto(); setStep((current+3)%5+1); });
  document.getElementById('pipe-next').addEventListener('click',function(){ stopAuto(); setStep(current%5+1); });
  dotsBox.addEventListener('click',function(e){
    var b=e.target.closest('button'); if(!b)return; stopAuto(); setStep(+b.dataset.step);
  });
  setStep(1);
  if(!REDUCED && io){
    var st=document.getElementById('pipe-stage');
    var pio=new IntersectionObserver(function(en){
      if(en[0].isIntersecting && !auto){ auto=setInterval(function(){ setStep(current%5+1); },6000); }
      else if(!en[0].isIntersecting) stopAuto();
    },{threshold:.4});
    pio.observe(st);
  }
})();

/* =====================================================================
   9 · FEDERATED ROUND simulator
===================================================================== */
(function(){
  var svg=document.getElementById('fr-svg'); if(!svg) return;
  var INK='#0F1830', MUT='#697691', SIG='#1BA5C4', VIO='#7C6BFF', ANOM='#FF6B4A', KEY='#B07E1F';
  var st={fedbn:true,secagg:false,dp:false,playing:false,round:1,pi:-1};

  var defs=s('defs',{},svg);
  var f=s('filter',{id:'fuzz',x:'-60%',y:'-60%',width:'220%',height:'220%'},defs);
  s('feGaussianBlur',{stdDeviation:'1.6'},f);
  var pat=s('pattern',{id:'maskpat',width:'6',height:'6',patternUnits:'userSpaceOnUse',patternTransform:'rotate(45)'},defs);
  s('rect',{width:'6',height:'6',fill:SIG},pat);
  s('rect',{width:'3',height:'6',fill:'#0F1830'},pat);

  /* coordinator */
  var srvPulse=s('circle',{cx:480,cy:62,r:54,fill:'rgba(27,165,196,.16)',opacity:0},svg);
  s('rect',{x:384,y:26,width:192,height:72,rx:14,fill:'#0f1b3e'},svg);
  txt(svg,480,56,'Coordinator',14.5,'#fff','middle',600);
  txt(svg,480,76,'sees updates, never data',10,'#9fb0d4','middle',400,true);
  var formula=txt(svg,480,120,'',12.5,VIO,'middle',700,true);

  /* organisations */
  var CL=[
    {x:36, name:'Org A · CICIDS'},
    {x:272,name:'Org B · UNSW'},
    {x:508,name:'Org C · ToN-IoT'},
    {x:744,name:'Org D · BoT-IoT'}
  ];
  var paths=[];
  CL.forEach(function(c,ci){
    c.cx=c.x+90;
    var gc=s('g',{},svg);
    s('rect',{x:c.x,y:268,width:180,height:140,rx:14,fill:'#fff',stroke:'#DCE3EF','stroke-width':1.5},gc);
    txt(gc,c.cx,290,c.name,12.5,INK,'middle',600);
    /* private traffic logs, behind a dashed fence with a padlock */
    c.fence=s('rect',{x:c.x+12,y:300,width:104,height:100,rx:10,fill:'rgba(27,165,196,.05)',stroke:SIG,'stroke-width':1.5,'stroke-dasharray':'5 4'},gc);
    s('path',{d:'M'+(c.x+17)+' 296 v-3 a4.5 4.5 0 0 1 9 0 v3',stroke:'#0f7d96','stroke-width':1.6,fill:'none'},gc);
    s('rect',{x:c.x+13,y:296,width:17,height:11,rx:3,fill:'#0f7d96'},gc);
    s('rect',{x:c.x+20,y:308,width:88,height:13,rx:3,fill:'#DCE3EF'},gc);
    txt(gc,c.x+25,318,'traffic logs',8,MUT,'start',500,true);
    for(var r=0;r<5;r++){
      var ry=327+r*14;
      s('rect',{x:c.x+20,y:ry,width:88,height:11,rx:3,fill:'#F5F7FB'},gc);
      s('rect',{x:c.x+25,y:ry+3.5,width:20,height:4,rx:2,fill:'#B9C6DE'},gc);
      s('rect',{x:c.x+50,y:ry+3.5,width:20,height:4,rx:2,fill:'#B9C6DE'},gc);
      s('rect',{x:c.x+75,y:ry+3.5,width:10+((ci*7+r*11)%18),height:4,rx:2,fill:'#8FD8E8'},gc);
    }
    c.hl=s('rect',{x:c.x+20,y:327,width:88,height:11,rx:3,fill:'rgba(124,107,255,.25)',opacity:0},gc);
    /* local model copy (tiny network) + training progress + BN block */
    s('rect',{x:c.x+124,y:302,width:44,height:52,rx:8,fill:'#F5F7FB',stroke:'#c9d3e4','stroke-width':1.2},gc);
    var nx=[c.x+133,c.x+146,c.x+159], l1=[312,324,336], l2=[318,330];
    l1.forEach(function(y1){ l2.forEach(function(y2){
      s('line',{x1:nx[0],y1:y1,x2:nx[1],y2:y2,stroke:'#c9d3e4','stroke-width':.8},gc);
      s('line',{x1:nx[1],y1:y2,x2:nx[2],y2:y1,stroke:'#c9d3e4','stroke-width':.8},gc);
    }); });
    l1.forEach(function(y){ s('circle',{cx:nx[0],cy:y,r:2.4,fill:VIO},gc); s('circle',{cx:nx[2],cy:y,r:2.4,fill:VIO},gc); });
    l2.forEach(function(y){ s('circle',{cx:nx[1],cy:y,r:2.4,fill:VIO},gc); });
    txt(gc,c.x+146,349,'model',8,MUT,'middle',500,true);
    s('rect',{x:c.x+124,y:362,width:44,height:6,rx:3,fill:'#EDF1F8'},gc);
    c.prog=s('rect',{x:c.x+124,y:362,width:0,height:6,rx:3,fill:VIO},gc);
    c.bnChip=s('rect',{x:c.x+124,y:376,width:44,height:22,rx:8,fill:'rgba(124,107,255,.12)',stroke:VIO,'stroke-width':1.4},gc);
    c.bnTxt=txt(gc,c.x+146,391,'BN',10,VIO,'middle',700,true);
    /* path to coordinator */
    paths.push(s('path',{d:'M'+c.cx+' 264 C '+c.cx+' 200, 480 180, 480 102',stroke:'#E4E9F3','stroke-width':1.6,fill:'none'},svg));
  });

  /* in-diagram step callouts (drawn under the moving packets) */
  var co=s('g',{},svg);
  function callout(x,y,lines,accent){
    var w=0; lines.forEach(function(L){ w=Math.max(w,L.length); });
    var bw=Math.min(w*6.4+26,470), bh=lines.length*16+14;
    var box=s('g',{},co);
    s('rect',{x:x-bw/2,y:y,width:bw,height:bh,rx:10,fill:'#fff',stroke:accent||'#DCE3EF','stroke-width':1.4,opacity:.97},box);
    lines.forEach(function(L,i){ txt(box,x,y+21+i*16,L,i===0?12:11.5,i===0?(accent||INK):'#38445F','middle',i===0?700:400); });
  }
  function clearCo(){ while(co.firstChild) co.removeChild(co.firstChild); }
  var hint=txt(svg,480,186,'▶ press Play to follow one training round, step by step',12.5,MUT,'middle',500);

  /* SecAgg key-exchange arcs between the organisations */
  function keyGlyph(){
    var kg=s('g',{opacity:0},svg);
    s('circle',{cx:-4,cy:0,r:3.4,fill:'#fff',stroke:KEY,'stroke-width':2},kg);
    s('path',{d:'M-0.6 0 H8 M5 0 v3.4 M8 0 v3.4',stroke:KEY,'stroke-width':2,fill:'none','stroke-linecap':'round'},kg);
    return kg;
  }
  var keyArcs=[[0,1],[1,2],[2,3],[0,3]].map(function(pr){
    var x0=CL[pr[0]].cx, x1=CL[pr[1]].cx, cy=(pr[1]-pr[0]===1)?216:156;
    var p=s('path',{d:'M'+x0+' 260 Q '+((x0+x1)/2)+' '+cy+' '+x1+' 260',stroke:KEY,'stroke-width':1.2,'stroke-dasharray':'3 5',fill:'none',opacity:0},svg);
    return {path:p,len:p.getTotalLength(),k1:keyGlyph(),k2:keyGlyph()};
  });
  function keysVis(v){ keyArcs.forEach(function(a){ a.path.setAttribute('opacity',v?.7:0); a.k1.setAttribute('opacity',v?1:0); a.k2.setAttribute('opacity',v?1:0); }); }

  /* packets = the model / its update, travelling the paths */
  var pk=CL.map(function(){
    var gp=s('g',{opacity:0},svg);
    var core=s('circle',{r:9,fill:VIO},gp);
    var hatch=s('circle',{r:9,fill:'url(#maskpat)',opacity:0},gp);
    var lock=s('path',{d:'M-3 -1 v-2 a3 3 0 0 1 6 0 v2 M-4 -1 h8 v6 h-8 z',stroke:'#fff','stroke-width':1.4,fill:'none',opacity:0},gp);
    var speck=s('g',{opacity:0},gp);
    for(var i=0;i<5;i++) s('circle',{cx:(i-2)*5,cy:(i%2?-8:8),r:1.6,fill:ANOM},speck);
    return {g:gp,core:core,hatch:hatch,lock:lock,speck:speck};
  });
  function pkSet(o){
    pk.forEach(function(p){
      p.core.setAttribute('fill',o.fill);
      p.hatch.setAttribute('opacity',o.hatch);
      p.lock.setAttribute('opacity',o.lock);
      p.speck.setAttribute('opacity',o.specks);
      if(o.blur) p.core.setAttribute('filter','url(#fuzz)'); else p.core.removeAttribute('filter');
    });
  }
  function place(i,t,up){ /* t 0..1 along path; path runs client->server */
    var pe=paths[i], len=pe.getTotalLength();
    var pt=pe.getPointAtLength((up?t:1-t)*len);
    pk[i].g.setAttribute('transform','translate('+pt.x+','+pt.y+')');
  }

  function bnPaint(){
    CL.forEach(function(c){
      c.bnChip.setAttribute('stroke', st.fedbn?VIO:'#B9C6DE');
      c.bnChip.setAttribute('fill', st.fedbn?'rgba(124,107,255,.12)':'#EDF1F8');
      c.bnTxt.setAttribute('fill', st.fedbn?VIO:MUT);
    });
  }
  bnPaint();

  /* phase list adapts to the toggles; each phase narrates itself on the diagram */
  function phases(){
    var ph=[{id:'send',d:2400},{id:'train',d:3000}];
    if(st.fedbn) ph.push({id:'fedbn',d:3400});
    if(st.secagg) ph.push({id:'keys',d:3000},{id:'mask',d:2400});
    if(st.dp) ph.push({id:'noise',d:2200});
    ph.push({id:'ret',d:2600},{id:'avg',d:3000});
    return ph;
  }
  var PH=phases();
  function stepNums(){
    var n={send:1,train:2}, k=3;
    if(st.fedbn){ n.fedbn=k; k++; }
    if(st.secagg){ n.keys=k; n.mask=k; k++; }
    if(st.dp){ n.noise=k; k++; }
    n.ret=k; k++; n.avg=k; n.total=k;
    return n;
  }
  var roundEl=document.getElementById('fr-round');
  function enterPhase(id){
    var n=stepNums(), T=' of '+n.total;
    clearCo(); formula.textContent=''; hint.setAttribute('opacity',0);
    srvPulse.setAttribute('opacity',0);
    keysVis(false);
    CL.forEach(function(c){ c.hl.setAttribute('opacity',0); c.bnChip.setAttribute('stroke-width',1.4); });
    if(id==='send'){
      pkSet({fill:VIO,hatch:0,lock:0,specks:0,blur:false});
      callout(480,166,['Step 1'+T+' · Send out the model',
        'The coordinator sends every organisation',
        'a copy of the shared model.'],SIG);
    }
    if(id==='train'){
      pk.forEach(function(p){ p.g.setAttribute('opacity',0); });
      callout(480,196,['Step 2'+T+' · Train at home',
        'Each organisation tunes its copy on its own',
        'traffic logs.'],SIG);
    }
    if(id==='fedbn'){
      pk.forEach(function(p){ p.g.setAttribute('opacity',0); });
      callout(480,150,['Step '+n.fedbn+T+' · FedBN: the scaling settings stay home',
        'Every network’s traffic runs at its own typical volume.',
        'The few settings that track that scale — the “BN” chip',
        'below — are left out of what gets sent back.'],VIO);
    }
    if(id==='keys'){
      pk.forEach(function(p){ p.g.setAttribute('opacity',0); });
      keysVis(true);
      callout(480,120,['Step '+n.keys+T+' · SecAgg+: swap secret keys',
        'Every pair of organisations exchanges a key',
        'that only those two know.'],KEY);
    }
    if(id==='mask'){
      pkSet({fill:VIO,hatch:0,lock:0,specks:0,blur:false});
      pk.forEach(function(p,i){ p.g.setAttribute('opacity',1); place(i,.04,true); });
      callout(480,120,['Step '+n.mask+T+' · SecAgg+: put on the masks',
        'Each update is hidden under masks built from the keys.',
        'Paired masks are equal and opposite: +m and −m.'],KEY);
    }
    if(id==='noise'){
      pkSet({fill:VIO,hatch:st.secagg?1:0,lock:st.secagg?1:0,specks:0,blur:false});
      pk.forEach(function(p,i){ p.g.setAttribute('opacity',1); place(i,.04,true); });
      callout(480,150,['Step '+n.noise+T+' · DP: add a pinch of noise',
        'Random noise is mixed into every update, so nothing',
        'about any single network can be read back out.'],'#E8462A');
    }
    if(id==='ret'){
      pkSet({fill:SIG,hatch:st.secagg?1:0,lock:st.secagg?1:0,specks:st.dp?1:0,blur:st.dp});
      var rlines=['Step '+n.ret+T+' · Send back only the update',
        'Only what the model learned travels back,',
        'never any traffic data.'];
      if(st.fedbn) rlines.push('(The BN scaling settings stay behind.)');
      callout(480,150,rlines,SIG);
    }
    if(id==='avg'){
      pk.forEach(function(p){ p.g.setAttribute('opacity',0); });
      formula.textContent='new model = (A + B + C + D) ÷ 4';
      var lines=['Step '+n.avg+T+' · FedAvg: Federated Averaging',
        'The coordinator averages the four updates into',
        'the next shared model, and the round repeats.'];
      if(st.secagg) lines.push('The masks cancel in the sum (+m − m = 0), so it',
        'only ever sees the total, never one update.');
      callout(480,146,lines,VIO);
    }
  }

  var t0=null, raf=null, curIdx=-1;
  function frame(now){
    if(!t0){ t0=now; PH=phases(); curIdx=-1; }
    var el=now-t0, idx=0, acc=0;
    while(idx<PH.length-1 && el>=acc+PH[idx].d){ acc+=PH[idx].d; idx++; }
    if(el>=acc+PH[idx].d){
      st.round=st.round%50+1; roundEl.textContent='Round '+st.round+' of 50';
      t0=now; PH=phases(); el=0; idx=0; acc=0; curIdx=-1;
    }
    var ph=PH[idx], t=(el-acc)/ph.d;
    if(idx!==curIdx){ curIdx=idx; enterPhase(ph.id); }
    if(ph.id==='send'){ pk.forEach(function(p,i){ p.g.setAttribute('opacity',1); place(i,Math.min(1,t*1.04),false); }); }
    if(ph.id==='train'){
      CL.forEach(function(c,i){
        c.prog.setAttribute('width',44*Math.min(1,t*1.1+i*.02));
        c.hl.setAttribute('opacity',.9);
        c.hl.setAttribute('y',327+(Math.floor(t*10)%5)*14);
      });
    }
    if(ph.id==='fedbn'){
      CL.forEach(function(c){ c.bnChip.setAttribute('stroke-width',1.4+1.8*Math.abs(Math.sin(t*6.28))); });
    }
    if(ph.id==='keys'){
      keyArcs.forEach(function(a){
        var p1=a.path.getPointAtLength(a.len*t), p2=a.path.getPointAtLength(a.len*(1-t));
        a.k1.setAttribute('transform','translate('+p1.x+','+p1.y+')');
        a.k2.setAttribute('transform','translate('+p2.x+','+p2.y+')');
      });
    }
    if(ph.id==='mask'){
      pk.forEach(function(p){ p.hatch.setAttribute('opacity',Math.min(1,t*1.5)); p.lock.setAttribute('opacity',t>.55?1:0); });
    }
    if(ph.id==='noise'){
      pk.forEach(function(p){ p.speck.setAttribute('opacity',.35+.65*Math.abs(Math.sin(t*9))); if(t>.15)p.core.setAttribute('filter','url(#fuzz)'); });
    }
    if(ph.id==='ret'){
      pk.forEach(function(p,i){ p.g.setAttribute('opacity',1); place(i,Math.min(1,t*1.04),true); });
      CL.forEach(function(c){ c.prog.setAttribute('width',0); });
    }
    if(ph.id==='avg'){
      srvPulse.setAttribute('opacity',.5+.5*Math.sin(t*6.28));
      srvPulse.setAttribute('r',54+9*t);
    }
    if(st.playing) raf=requestAnimationFrame(frame);
  }
  var play=document.getElementById('fr-play');
  function setPlaying(v){
    st.playing=v; play.textContent=v?'❚❚ Pause':'▶ Play'; play.setAttribute('aria-pressed',v);
    if(v){ t0=null; raf=requestAnimationFrame(frame); }
    else if(raf) cancelAnimationFrame(raf);
  }
  play.addEventListener('click',function(){
    if(REDUCED){ /* step through the phases statically */
      PH=phases();
      st.pi=(st.pi+1)%PH.length;
      var id=PH[st.pi].id;
      enterPhase(id);
      pk.forEach(function(p,i){
        var show=(id==='send'||id==='ret'||id==='mask'||id==='noise');
        p.g.setAttribute('opacity',show?1:0);
        if(show) place(i,(id==='send'||id==='ret')?.55:.04,id!=='send');
      });
      if(id==='mask') pk.forEach(function(p){ p.hatch.setAttribute('opacity',1); p.lock.setAttribute('opacity',1); });
      if(id==='noise') pk.forEach(function(p){ p.speck.setAttribute('opacity',1); });
      if(id==='keys') keyArcs.forEach(function(a){
        var m=a.path.getPointAtLength(a.len*.5);
        a.k1.setAttribute('transform','translate('+(m.x-8)+','+m.y+')');
        a.k2.setAttribute('transform','translate('+(m.x+8)+','+m.y+')');
      });
      if(id==='train') CL.forEach(function(c){ c.prog.setAttribute('width',26); c.hl.setAttribute('opacity',.9); });
      if(id==='fedbn') CL.forEach(function(c){ c.bnChip.setAttribute('stroke-width',3); });
      if(id==='avg'){ srvPulse.setAttribute('opacity',.6); st.round=st.round%50+1; roundEl.textContent='Round '+st.round+' of 50'; }
      return;
    }
    setPlaying(!st.playing);
  });
  [['tg-fedbn','fedbn'],['tg-secagg','secagg'],['tg-dp','dp']].forEach(function(cfg){
    var b=document.getElementById(cfg[0]);
    b.addEventListener('click',function(){
      st[cfg[1]]=!st[cfg[1]];
      b.classList.toggle('on',st[cfg[1]]); b.setAttribute('aria-pressed',st[cfg[1]]);
      bnPaint();
      t0=null; st.pi=-1; /* restart the round so the new option plays from the top */
    });
  });
  /* autoplay when scrolled into view; pause offscreen */
  if(!REDUCED && io){
    var fio=new IntersectionObserver(function(en){
      if(en[0].isIntersecting && !st.playing) setPlaying(true);
      else if(!en[0].isIntersecting && st.playing) setPlaying(false);
    },{threshold:.35});
    fio.observe(document.getElementById('fedround'));
  }
})();
})();

/* ---- Hero NetFlow graph: nodes = IPs, edges = flows, occasional anomaly flare ---- */
(function(){
  var prefersReduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var canvas = document.getElementById('flowcanvas');
  if(!canvas) return;
  var ctx = canvas.getContext('2d');
  var W,H,DPR,nodes=[],edges=[],pulses=[];
  var SIGNAL='63,211,238', ANOM='255,107,74', VIOLET='150,135,255';

  function rand(a,b){return a+Math.random()*(b-a);}

  function build(){
    DPR = Math.min(window.devicePixelRatio||1, 2);
    W = canvas.clientWidth; H = canvas.clientHeight;
    canvas.width = W*DPR; canvas.height = H*DPR;
    ctx.setTransform(DPR,0,0,DPR,0,0);
    nodes=[]; edges=[]; pulses=[];
    var count = Math.max(14, Math.min(30, Math.floor(W*H/42000)));
    for(var i=0;i<count;i++){
      nodes.push({
        x:rand(0.04,0.97)*W, y:rand(0.05,0.95)*H,
        r:rand(1.6,3.4),
        vx:rand(-.06,.06), vy:rand(-.06,.06),
        hub: Math.random()<0.22
      });
    }
    // connect: each node to a couple of nearest others
    for(var i=0;i<nodes.length;i++){
      var d=[];
      for(var j=0;j<nodes.length;j++) if(i!==j){
        var dx=nodes[i].x-nodes[j].x, dy=nodes[i].y-nodes[j].y;
        d.push({j:j, dist:dx*dx+dy*dy});
      }
      d.sort(function(a,b){return a.dist-b.dist;});
      var k = nodes[i].hub?4:2;
      for(var m=0;m<k && m<d.length;m++){
        var jj=d[m].j;
        if(!edges.some(function(e){return (e.a===jj&&e.b===i)||(e.a===i&&e.b===jj);}))
          edges.push({a:i,b:jj});
      }
    }
  }

  function spawnPulse(){
    if(!edges.length) return;
    var e = edges[Math.floor(Math.random()*edges.length)];
    var anom = Math.random()<0.16;
    pulses.push({e:e, t:0, speed:rand(0.006,0.013), anom:anom, dir:Math.random()<.5?1:-1});
  }

  var lastSpawn=0;
  function frame(now){
    ctx.clearRect(0,0,W,H);
    // drift nodes
    for(var i=0;i<nodes.length;i++){
      var n=nodes[i];
      n.x+=n.vx; n.y+=n.vy;
      if(n.x<8||n.x>W-8) n.vx*=-1;
      if(n.y<8||n.y>H-8) n.vy*=-1;
    }
    // edges
    ctx.lineWidth=1;
    for(var i=0;i<edges.length;i++){
      var a=nodes[edges[i].a], b=nodes[edges[i].b];
      ctx.strokeStyle='rgba('+SIGNAL+',0.10)';
      ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
    }
    // nodes
    for(var i=0;i<nodes.length;i++){
      var n=nodes[i];
      ctx.beginPath();
      ctx.fillStyle = n.hub? 'rgba('+VIOLET+',0.9)' : 'rgba('+SIGNAL+',0.75)';
      ctx.arc(n.x,n.y,n.r,0,6.2832); ctx.fill();
      if(n.hub){ ctx.beginPath(); ctx.strokeStyle='rgba('+VIOLET+',0.25)'; ctx.lineWidth=1;
        ctx.arc(n.x,n.y,n.r+4,0,6.2832); ctx.stroke(); }
    }
    // spawn
    if(now-lastSpawn > 240){ spawnPulse(); lastSpawn=now; }
    // pulses (flows travelling along an edge)
    for(var i=pulses.length-1;i>=0;i--){
      var p=pulses[i]; p.t+=p.speed;
      if(p.t>=1){ pulses.splice(i,1); continue; }
      var a=nodes[p.e.a], b=nodes[p.e.b];
      var t = p.dir>0? p.t : 1-p.t;
      var x=a.x+(b.x-a.x)*t, y=a.y+(b.y-a.y)*t;
      var col = p.anom?ANOM:SIGNAL;
      // trailing glow along segment
      var grad=ctx.createLinearGradient(a.x,a.y,b.x,b.y);
      grad.addColorStop(Math.max(0,t-0.12),'rgba('+col+',0)');
      grad.addColorStop(t,'rgba('+col+','+(p.anom?0.55:0.4)+')');
      grad.addColorStop(Math.min(1,t+0.01),'rgba('+col+',0)');
      ctx.strokeStyle=grad; ctx.lineWidth=p.anom?2.2:1.4;
      ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
      // moving head
      ctx.beginPath(); ctx.fillStyle='rgba('+col+',0.95)';
      ctx.arc(x,y,p.anom?3.2:2,0,6.2832); ctx.fill();
      if(p.anom){
        ctx.beginPath(); ctx.strokeStyle='rgba('+ANOM+','+(0.5*(1-p.t))+')'; ctx.lineWidth=1.5;
        ctx.arc(x,y,3.2+10*p.t,0,6.2832); ctx.stroke();
      }
    }
    raf=requestAnimationFrame(frame);
  }

  var raf;
  function start(){ cancelAnimationFrame(raf); build();
    if(prefersReduced){ // draw one static frame
      ctx.clearRect(0,0,W,H);
      for(var i=0;i<edges.length;i++){var a=nodes[edges[i].a],b=nodes[edges[i].b];
        ctx.strokeStyle='rgba('+SIGNAL+',0.10)';ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();}
      for(var i=0;i<nodes.length;i++){var n=nodes[i];ctx.beginPath();
        ctx.fillStyle=n.hub?'rgba('+VIOLET+',0.9)':'rgba('+SIGNAL+',0.75)';ctx.arc(n.x,n.y,n.r,0,6.2832);ctx.fill();}
      return;
    }
    raf=requestAnimationFrame(frame);
  }
  var rt;
  window.addEventListener('resize',function(){clearTimeout(rt);rt=setTimeout(start,180);});
  start();
})();

/* ---- Problem → Solution flow map: animated connectors + hover tracing ---- */
(function(){
  var map=document.getElementById('probmap'); if(!map) return;
  var REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var SVGNS='http://www.w3.org/2000/svg';
  var svg=map.querySelector('.ob-links');
  var obs=[].slice.call(map.querySelectorAll('.ob'));
  var fixes={fixA:document.getElementById('fixA'),fixB:document.getElementById('fixB')};
  var COL={fixA:'#1BA5C4',fixB:'#7C6BFF'};
  var links=[], revealed=false, running=false, raf=null;

  function build(){
    while(svg.firstChild) svg.removeChild(svg.firstChild);
    links=[];
    if(getComputedStyle(svg).display==='none') return;
    var mb=map.getBoundingClientRect();
    svg.setAttribute('viewBox','0 0 '+mb.width+' '+mb.height);
    var counts={fixA:0,fixB:0}, idx={fixA:0,fixB:0};
    obs.forEach(function(o){ counts[o.dataset.to]++; });
    obs.forEach(function(o){
      var to=o.dataset.to, f=fixes[to];
      var ob=o.getBoundingClientRect(), fb=f.getBoundingClientRect();
      var x0=ob.left+ob.width/2-mb.left, y0=ob.bottom-mb.top+4;
      var i=idx[to]++, n=counts[to];
      var x1=fb.left-mb.left+fb.width*((i+1)/(n+1)), y1=fb.top-mb.top-3;
      var p=document.createElementNS(SVGNS,'path');
      p.setAttribute('d','M'+x0+' '+y0+' C '+x0+' '+(y0+58)+', '+x1+' '+(y1-58)+', '+x1+' '+y1);
      p.setAttribute('fill','none'); p.setAttribute('stroke',COL[to]);
      p.setAttribute('stroke-width','1.8'); p.setAttribute('stroke-linecap','round');
      p.setAttribute('opacity','.4');
      svg.appendChild(p);
      var dot=document.createElementNS(SVGNS,'circle');
      dot.setAttribute('r','3'); dot.setAttribute('fill',COL[to]); dot.setAttribute('opacity','0');
      svg.appendChild(dot);
      links.push({ob:o,to:to,path:p,dot:dot,len:p.getTotalLength(),t:Math.random()});
    });
  }
  function drawIn(){
    if(REDUCED) return;
    links.forEach(function(l,i){
      l.path.style.transition='none';
      l.path.style.strokeDasharray=l.len; l.path.style.strokeDashoffset=l.len;
      void l.path.getBoundingClientRect();
      l.path.style.transition='stroke-dashoffset .9s ease '+(i*.1)+'s';
      l.path.style.strokeDashoffset=0;
    });
  }
  function tick(){
    links.forEach(function(l){
      l.t+=.0045; if(l.t>1) l.t-=1;
      var pt=l.path.getPointAtLength(l.t*l.len);
      l.dot.setAttribute('cx',pt.x); l.dot.setAttribute('cy',pt.y);
      l.dot.setAttribute('opacity',.85);
    });
    if(running) raf=requestAnimationFrame(tick);
  }
  function setRunning(v){
    if(REDUCED||!links.length) return;
    if(v&&!running){ running=true; raf=requestAnimationFrame(tick); }
    else if(!v&&running){ running=false; cancelAnimationFrame(raf); }
  }
  function clear(){
    obs.forEach(function(o){ o.classList.remove('hot'); });
    fixes.fixA.classList.remove('hot'); fixes.fixB.classList.remove('hot');
    links.forEach(function(l){ l.path.setAttribute('opacity',.4); l.path.setAttribute('stroke-width',1.8); });
  }
  function hotTo(to,obOnly){
    clear(); fixes[to].classList.add('hot');
    obs.forEach(function(o){ if(o.dataset.to===to&&(!obOnly||o===obOnly)) o.classList.add('hot'); });
    links.forEach(function(l){
      var on=l.to===to&&(!obOnly||l.ob===obOnly);
      l.path.setAttribute('opacity',on?.95:.12);
      l.path.setAttribute('stroke-width',on?2.6:1.8);
    });
  }
  obs.forEach(function(o){
    o.addEventListener('mouseenter',function(){ hotTo(o.dataset.to,o); });
    o.addEventListener('focus',function(){ hotTo(o.dataset.to,o); });
  });
  ['fixA','fixB'].forEach(function(id){
    fixes[id].addEventListener('mouseenter',function(){ hotTo(id); });
    fixes[id].addEventListener('focus',function(){ hotTo(id); });
  });
  map.addEventListener('mouseleave',clear);

  var rt; window.addEventListener('resize',function(){
    clearTimeout(rt);
    rt=setTimeout(function(){ if(revealed){ build(); setRunning(true); } },180);
  });
  if('IntersectionObserver' in window && !REDUCED){
    var pio=new IntersectionObserver(function(en){
      en.forEach(function(e){
        if(e.isIntersecting){
          if(!revealed){
            revealed=true; map.classList.add('in');
            /* wait for the tiles' entrance transition to settle, then wire them up */
            setTimeout(function(){ build(); drawIn(); setRunning(true); },620);
          } else setRunning(true);
        } else setRunning(false);
      });
    },{threshold:.2});
    pio.observe(map);
  } else {
    revealed=true; map.classList.add('in'); build();
  }
})();

/**
 * graph-render.js — modality-resolved knowledge graph for the Graph tab.
 *
 * Loads AFTER api.js and d3. Exposes renderVGIAGraph(videoId, containerEl),
 * which fetches /graph/video/{id} (get_video_graph), draws ALL nodes with
 * physics auto-scaled to the node count, colors edges/nodes by source modality,
 * and offers a cross-modal filter. Clicking a node jumps the YouTube player to
 * that entity's first mention (uses the existing seekVideoTo()).
 *
 * Requires d3 (already loaded on the page).
 */
(function () {
  'use strict';

  const MOD = { speech: '#378ADD', visual: '#BA7517', mixed: '#1D9E75', unknown: '#888780' };
  const MODLABEL = { speech: 'spoken', visual: 'on-screen', mixed: 'both (cross-modal)', unknown: 'unresolved' };

  function isDark() {
    return document.documentElement.dataset.theme === 'dark'
        || window.matchMedia('(prefers-color-scheme: dark)').matches;
  }

  // main entry
  window.renderVGIAGraph = async function (videoId, container) {
    if (!videoId || !container) return;
    if (typeof d3 === 'undefined') { container.innerHTML = '<p style="padding:20px;color:var(--text-muted)">graph library not loaded</p>'; return; }

    container.innerHTML =
      '<div style="padding:28px;text-align:center;color:var(--text-muted);font-size:14px">Loading knowledge graph…</div>';

    let data;
    try {
      const r = await fetch('/graph/video/' + encodeURIComponent(videoId));
      data = await r.json();
    } catch (e) {
      container.innerHTML = '<div style="padding:28px;text-align:center;color:var(--text-muted)">Couldn\'t load the graph.</div>';
      return;
    }

    const rawNodes = (data && data.nodes) || [];
    const rawEdges = (data && data.edges) || [];
    if (!rawNodes.length) {
      container.innerHTML = '<div style="padding:34px;text-align:center;color:var(--text-muted);font-size:14px">No knowledge graph was extracted for this video.</div>';
      return;
    }

    const dark = isDark();
    const line = dark ? '#34342f' : '#dcdbd3';
    const ink = dark ? '#c8c7bc' : '#44443f';
    const nodeStroke = dark ? '#15151a' : '#fbfbf9';

    // ---- build UI shell (filter pills + svg + detail line) ----
    container.innerHTML = ''
      + '<div id="vg-pills" style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 12px">'
      +   '<span style="font-size:13px;color:var(--text-muted);margin-right:2px">source</span>'
      +   pill('all', 'all facts', null, true)
      +   pill('speech', 'spoken', MOD.speech)
      +   pill('visual', 'on-screen', MOD.visual)
      +   pill('mixed', 'both ↔ cross-modal', MOD.mixed)
      +   '<span id="vg-count" style="margin-left:auto;font-size:12px;color:var(--text-faint)"></span>'
      + '</div>'
      + '<div style="position:relative;background:var(--bg-card,rgba(0,0,0,.02));border-radius:12px;overflow:hidden;border:1px solid var(--border-color,transparent)">'
      +   '<svg id="vg-svg" width="100%" style="display:block"></svg>'
      +   '<div id="vg-tip" style="position:absolute;pointer-events:none;opacity:0;background:var(--bg-card,#1a1a1a);border:1px solid var(--border-light,#333);border-radius:8px;padding:8px 10px;font-size:12px;max-width:230px;line-height:1.5;transition:opacity .12s;z-index:5"></div>'
      + '</div>'
      + '<div id="vg-det" style="margin-top:10px;font-size:13px;color:var(--text-secondary);min-height:32px;line-height:1.6"></div>';

    function pill(m, label, color, on) {
      const dot = color ? '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:' + color + ';margin-right:6px"></span>' : '';
      return '<button class="vg-pill' + (on ? ' on' : '') + '" data-m="' + m + '" '
        + 'style="font-size:13px;padding:5px 13px;border-radius:99px;cursor:pointer;'
        + 'border:1px solid var(--border-light,#444);background:' + (on ? 'var(--brand-tint,rgba(120,120,255,.15))' : 'transparent') + ';color:var(--text-primary,#eee)">'
        + dot + label + '</button>';
    }

    // ---- dimensions scale with node count ----
    const n = rawNodes.length;
    const W = 700;
    const H = Math.max(460, Math.min(900, 380 + n * 1.1));
    const svg = d3.select('#vg-svg').attr('viewBox', '0 0 ' + W + ' ' + H);
    const root = svg.append('g');

    document.getElementById('vg-count').textContent = n + ' entities · ' + rawEdges.length + ' relations';

    // ---- data ----
    const nodes = rawNodes.map(d => ({
      id: d.id, name: (d.name || d.id || '').slice(0, 28),
      type: d.type || 'OTHER', mentions: d.mentions || d.m || 0
    }));
    const byId = new Map(nodes.map(d => [d.id, d]));
    // node modality = majority modality of its incident edges
    const nmod = {};
    rawEdges.forEach(e => {
      const m = e.modality || 'unknown';
      [e.source, e.target].forEach(id => {
        nmod[id] = nmod[id] || { speech: 0, visual: 0, mixed: 0, unknown: 0 };
        nmod[id][m] = (nmod[id][m] || 0) + 1;
      });
    });
    nodes.forEach(d => {
      const c = nmod[d.id] || {};
      d.mod = Object.keys(c).sort((a, b) => c[b] - c[a])[0] || 'unknown';
    });

    const links = rawEdges
      .filter(e => byId.has(e.source) && byId.has(e.target))
      .map(e => ({ source: e.source, target: e.target, relation: e.relation,
                   modality: e.modality || 'unknown', timestamp: e.timestamp,
                   speaker: e.speaker, snippet: e.snippet }));

    const deg = {};
    links.forEach(l => { deg[l.source] = (deg[l.source] || 0) + 1; deg[l.target] = (deg[l.target] || 0) + 1; });
    const maxDeg = d3.max(nodes, d => deg[d.id] || 1) || 1;
    // smaller nodes when there are many
    const rMax = n > 200 ? 13 : n > 80 ? 17 : 22;
    const R = d3.scaleSqrt().domain([1, maxDeg]).range([3.2, rMax]);

    const L = root.append('g').selectAll('line').data(links).join('line')
      .attr('stroke', d => MOD[d.modality] || line)
      .attr('stroke-opacity', d => d.modality === 'mixed' ? 0.75 : 0.28)
      .attr('stroke-width', d => d.modality === 'mixed' ? 2 : 1)
      .attr('stroke-linecap', 'round');

    const N = root.append('g').selectAll('g').data(nodes).join('g').style('cursor', 'pointer');
    N.append('circle').attr('r', d => R(deg[d.id] || 1))
      .attr('fill', d => MOD[d.mod]).attr('fill-opacity', 0.92)
      .attr('stroke', nodeStroke).attr('stroke-width', 1.5);
    // label only prominent nodes so a big graph stays readable
    const labelThresh = n > 200 ? Math.max(6, maxDeg * 0.18) : n > 80 ? 4 : 2;
    N.append('text').text(d => d.name)
      .attr('font-size', 11).attr('fill', ink).attr('text-anchor', 'middle')
      .attr('dy', d => -R(deg[d.id] || 1) - 5)
      .attr('font-family', 'Plus Jakarta Sans, system-ui, sans-serif')
      .style('pointer-events', 'none')
      .attr('opacity', d => (deg[d.id] || 0) >= labelThresh ? 1 : 0);

    // physics tuned to size
    const charge = n > 200 ? -90 : n > 80 ? -180 : -320;
    const dist = n > 200 ? 34 : n > 80 ? 50 : 74;
    const sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(dist).strength(0.28))
      .force('charge', d3.forceManyBody().strength(charge))
      .force('center', d3.forceCenter(W / 2, H / 2))
      .force('collide', d3.forceCollide().radius(d => R(deg[d.id] || 1) + (n > 200 ? 3 : 8)))
      .on('tick', () => {
        L.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
         .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        N.attr('transform', d => 'translate(' +
          Math.max(16, Math.min(W - 16, d.x)) + ',' +
          Math.max(16, Math.min(H - 16, d.y)) + ')');
      });
    // cool big graphs faster so they settle
    if (n > 200) sim.alphaDecay(0.035);

    svg.call(d3.zoom().scaleExtent([0.3, 5]).on('zoom', e => root.attr('transform', e.transform)));
    N.call(d3.drag()
      .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }));

    // neighbours
    const nb = new Map();
    links.forEach(l => {
      const a = l.source.id || l.source, b = l.target.id || l.target;
      if (!nb.has(a)) nb.set(a, new Set()); if (!nb.has(b)) nb.set(b, new Set());
      nb.get(a).add(b); nb.get(b).add(a);
    });

    const tip = document.getElementById('vg-tip');
    const det = document.getElementById('vg-det');
    let CUR = 'all';

    N.on('mouseenter', function (ev, d) {
      const s = nb.get(d.id) || new Set();
      N.attr('opacity', x => x.id === d.id || s.has(x.id) ? 1 : 0.14);
      L.attr('stroke-opacity', l => {
        const a = l.source.id || l.source, b = l.target.id || l.target;
        return a === d.id || b === d.id ? 0.9 : 0.04;
      });
      tip.style.opacity = 1;
      tip.innerHTML = '<b>' + d.name + '</b><br><span style="color:' + MOD[d.mod] + '">● '
        + MODLABEL[d.mod] + '</span> · ' + (deg[d.id] || 0) + ' links · ' + d.mentions + ' mentions';
      const r = svg.node().getBoundingClientRect(), sc = r.width / W;
      tip.style.left = Math.min(d.x * sc + 14, r.width - 240) + 'px';
      tip.style.top = (d.y * sc - 6) + 'px';
    }).on('mouseleave', function () { applyFilter(CUR); tip.style.opacity = 0; })
    .on('click', function (ev, d) {
      const es = links.filter(l => (l.source.id || l.source) === d.id || (l.target.id || l.target) === d.id);
      const withTs = es.filter(l => l.timestamp != null).sort((a, b) => a.timestamp - b.timestamp);
      const mc = es.filter(l => l.modality === 'mixed').length;
      det.innerHTML = '<b>' + d.name + '</b> — ' + es.length + ' relations'
        + (mc ? ' · <span style="color:#1D9E75">' + mc + ' confirmed by both speech and on-screen text</span>' : '')
        + (withTs.length ? ' · <span style="color:var(--brand-light,#a78bfa)">jumping to first mention…</span>' : '');
      if (withTs.length && typeof window.seekVideoTo === 'function') {
        window.seekVideoTo(withTs[0].timestamp);
      }
    });

    function applyFilter(m) {
      CUR = m;
      L.attr('display', l => m === 'all' || l.modality === m ? null : 'none')
       .attr('stroke-opacity', l => m === 'all' ? (l.modality === 'mixed' ? 0.75 : 0.28) : (l.modality === m ? 0.85 : 0));
      const active = new Set();
      links.forEach(l => {
        if (m === 'all' || l.modality === m) {
          active.add(l.source.id || l.source); active.add(l.target.id || l.target);
        }
      });
      N.attr('opacity', d => m === 'all' || active.has(d.id) ? 1 : 0.1);
      if (m === 'mixed') {
        det.innerHTML = '<b style="color:#1D9E75">Cross-modal facts</b> — confirmed by BOTH the spoken transcript and the on-screen text. A text-only graph can\'t express this; the modality tag makes it queryable.';
      } else if (m === 'all') { det.innerHTML = ''; }
      else { det.innerHTML = 'Showing facts sourced from <span style="color:' + MOD[m] + '">' + MODLABEL[m] + '</span> only.'; }
    }

    document.querySelectorAll('.vg-pill').forEach(b => b.addEventListener('click', () => {
      document.querySelectorAll('.vg-pill').forEach(x => {
        x.classList.remove('on'); x.style.background = 'transparent';
      });
      b.classList.add('on');
      b.style.background = 'var(--brand-tint,rgba(120,120,255,.15))';
      applyFilter(b.dataset.m);
    }));
  };
})();

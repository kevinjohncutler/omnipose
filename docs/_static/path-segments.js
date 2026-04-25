document.addEventListener("DOMContentLoaded", () => {
  const nodes = document.querySelectorAll(
    ".sig-prename.descclassname > span.pre"
  );

  const sinebow = (t) => {
    // True sinebow: full hue sweep across 0..1
    const r = 0.5 + 0.5 * Math.sin(2 * Math.PI * (t + 0 / 3));
    const g = 0.5 + 0.5 * Math.sin(2 * Math.PI * (t + 1 / 3));
    const b = 0.5 + 0.5 * Math.sin(2 * Math.PI * (t + 2 / 3));
    const to255 = (v) => Math.round(v * 255);
    return `rgb(${to255(r)}, ${to255(g)}, ${to255(b)})`;
  };

  const debug = new URLSearchParams(window.location.search).has("color-debug");
  const debugRows = [];

  nodes.forEach((node) => {
    const text = node.textContent || "";
    const segments = text.split(".").filter(Boolean);
    if (segments.length <= 1) {
      return;
    }

    const frag = document.createDocumentFragment();
    const count = segments.length + 1;

    segments.forEach((seg, idx) => {
      const span = document.createElement("span");
      span.className = "omnipose-path-seg";
      span.style.color = sinebow(idx / count);
      span.textContent = seg;
      frag.appendChild(span);

      const dot = document.createElement("span");
      dot.className = "omnipose-path-dot";
      dot.textContent = ".";
      frag.appendChild(dot);
    });

    node.textContent = "";
    node.appendChild(frag);

    const sigName = node.closest("dt")?.querySelector(".sig-name.descname");
    if (sigName) {
      const color = sinebow(segments.length / count);
      sigName.style.setProperty("color", color, "important");
      sigName.querySelectorAll(".pre").forEach((child) => {
        child.style.setProperty("color", color, "important");
      });

      if (debug) {
        const pre = sigName.querySelector(".pre");
        const colorValue = pre
          ? getComputedStyle(pre).color
          : getComputedStyle(sigName).color;
        debugRows.push(`${text}${pre ? pre.textContent || "" : ""} -> ${colorValue}`);
      }
    }
  });

  if (debug && debugRows.length) {
    const panel = document.createElement("div");
    panel.className = "omnipose-color-debug";
    panel.textContent = debugRows.slice(0, 12).join("\n");
    document.body.appendChild(panel);
  }
});

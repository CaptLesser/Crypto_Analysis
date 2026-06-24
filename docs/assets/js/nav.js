document.documentElement.classList.remove("no-js");

const nav = document.querySelector("[data-nav]");
const toggle = document.querySelector("[data-nav-toggle]");

if (nav && toggle) {
  toggle.addEventListener("click", () => {
    const isOpen = nav.dataset.open === "true";
    nav.dataset.open = String(!isOpen);
    toggle.setAttribute("aria-expanded", String(!isOpen));
  });
}

const contentsRails = document.querySelectorAll("[data-doc-rail-shell]");
const narrowViewport = window.matchMedia("(max-width: 760px)");

contentsRails.forEach((shell, index) => {
  const rail = shell.querySelector("[data-doc-rail]");
  if (!(rail instanceof HTMLElement)) {
    return;
  }
  const railToggle = rail.querySelector("[data-doc-rail-toggle]");
  const edgeToggle = shell.querySelector("[data-doc-rail-edge-toggle]");
  const railContent = rail.querySelector("[data-doc-rail-content]");
  const railLabel = rail.querySelector("[data-doc-rail-toggle-label]");
  const layout = shell.closest(".doc-layout");
  let userChanged = false;

  if (
    !(railToggle instanceof HTMLButtonElement) ||
    !(edgeToggle instanceof HTMLButtonElement) ||
    !(railContent instanceof HTMLElement)
  ) {
    return;
  }

  if (!railContent.id) {
    railContent.id = `doc-contents-${index + 1}`;
    railToggle.setAttribute("aria-controls", railContent.id);
    edgeToggle.setAttribute("aria-controls", railContent.id);
  }

  const setExpanded = (expanded) => {
    railToggle.setAttribute("aria-expanded", String(expanded));
    railToggle.setAttribute("aria-label", expanded ? "Hide contents" : "Show contents");
    edgeToggle.setAttribute("aria-expanded", String(expanded));
    edgeToggle.setAttribute("aria-label", expanded ? "Hide contents" : "Show contents");
    rail.dataset.collapsed = String(!expanded);
    shell.dataset.collapsed = String(!expanded);
    rail.hidden = !expanded;
    edgeToggle.hidden = expanded;
    railContent.hidden = !expanded;
    if (layout instanceof HTMLElement) {
      layout.dataset.railCollapsed = String(!expanded);
    }
    if (railLabel) {
      railLabel.textContent = expanded ? "Hide" : "Show";
    }
  };

  setExpanded(!narrowViewport.matches);

  railToggle.addEventListener("click", () => {
    userChanged = true;
    const expanded = railToggle.getAttribute("aria-expanded") === "true";
    setExpanded(!expanded);
  });

  edgeToggle.addEventListener("click", () => {
    userChanged = true;
    setExpanded(true);
  });

  narrowViewport.addEventListener("change", (event) => {
    if (!userChanged) {
      setExpanded(!event.matches);
    }
  });
});

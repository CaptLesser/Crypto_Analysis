const canTransition = "startViewTransition" in document;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

document.documentElement.dataset.viewTransitions = String(canTransition && !reduceMotion);

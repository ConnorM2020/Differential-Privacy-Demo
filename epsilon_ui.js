// epsilon_ui.js

// This script dynamically explains epsilon selection with visual feedback.
document.addEventListener("DOMContentLoaded", function () {
  const epsilonInput = document.querySelector("input[name='epsilon_count']");
  const epsilonInfo = document.createElement("div");

  epsilonInfo.id = "epsilon-feedback";
  epsilonInfo.style.marginTop = "10px";
  epsilonInput.parentNode.insertBefore(epsilonInfo, epsilonInput.nextSibling);

  epsilonInput.addEventListener("input", function () {
    const val = parseFloat(epsilonInput.value);
    let message = "";
    if (val < 0.2) {
      message = "Very private, but high noise";
    } else if (val < 1.0) {
      message = "Balanced privacy and accuracy";
    } else if (val >= 1.0) {
      message = "More accurate, but weaker privacy";
    } else {
      message = "";
    }
    epsilonInfo.textContent = `ε = ${val.toFixed(2)} → ${message}`;
  });
});

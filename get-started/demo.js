const audioPlayer = document.querySelector("#audio-player audio");
const container = document.getElementById("transcript-container");

const wordMap = [];
const alignmentMap = [];
let prevWord = null;
let prevAlignment = null;

fetch("../_assets/taleoftwocities_01_dickens_64kb_trimmed.json")
  .then((r) => r.json())
  .then((data) => {
    data.speeches.forEach((speech) => {
      let para = document.createElement("p");
      para.className = "chunk";

      speech.alignments.forEach((alignment) => {
        const sentenceSpan = document.createElement("span");
        sentenceSpan.className = "alignment";

        // Click sentence to jump audio
        sentenceSpan.addEventListener("click", () => {
          audioPlayer.currentTime = alignment.start;
          audioPlayer.play();
        });

        alignment.words.forEach((word) => {
          const wordSpan = document.createElement("span");
          wordSpan.className = "word";
          wordSpan.textContent = word.text;
          wordSpan.dataset.start = word.start;
          wordSpan.dataset.end = word.end;
          sentenceSpan.appendChild(wordSpan);

          wordMap.push({ el: wordSpan, start: word.start, end: word.end });
        });

        para.appendChild(sentenceSpan);
        alignmentMap.push({
          el: sentenceSpan,
          start: alignment.start,
          end: alignment.end,
        });

        // No trailing whitespace signals a paragraph break
        if (!alignment.text.endsWith(" ")) {
          container.appendChild(para);
          para = document.createElement("p");
          para.className = "chunk";
        }
      });

      // Append any remaining sentences
      if (para.childElementCount > 0) {
        container.appendChild(para);
      }
    });
  });

function updateHighlight() {
  const t = audioPlayer.currentTime;

  const curWord = wordMap.find((w) => t >= w.start && t < w.end);
  if (curWord && curWord.el !== prevWord) {
    if (prevWord) prevWord.classList.remove("highlight-word");
    curWord.el.classList.add("highlight-word");
    prevWord = curWord.el;
  }

  const curAlignment = alignmentMap.find((a) => t >= a.start && t < a.end);
  if (curAlignment && curAlignment.el !== prevAlignment) {
    if (prevAlignment) prevAlignment.classList.remove("highlight-sentence");
    curAlignment.el.classList.add("highlight-sentence");
    prevAlignment = curAlignment.el;
  }
}

// Update on seek (dragging progress bar while paused)
audioPlayer.addEventListener("seeked", updateHighlight);

// Use requestAnimationFrame (~60fps) instead of timeupdate (~4fps)
// so short words (< 250ms) don't get skipped
function tick() {
  if (!audioPlayer.paused) {
    updateHighlight();
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

/**
 * Interactive transcript player.
 * Adapted from docs/get-started/demo.js for the search app.
 */
(function () {
  const audioPlayer = document.querySelector("#audio-player audio");
  const container = document.getElementById("transcript-container");
  if (!audioPlayer || !container) return;

  const documentId = container.dataset.documentId;
  const seekTime = parseFloat(container.dataset.seekTime) || 0;
  const searchQuery = container.dataset.query || "";

  const wordMap = [];
  const alignmentMap = [];
  let prevWord = null;
  let prevAlignment = null;

  container.innerHTML = '<div class="transcript-loading">Loading transcript...</div>';

  // Seek and play immediately — don't wait for transcript fetch
  if (seekTime > 0) {
    audioPlayer.currentTime = seekTime;
    var p = audioPlayer.play();
    if (p) {
      p.catch(function () {
        // Browser blocked autoplay — show a play button overlay
        var card = document.getElementById("audio-player");
        var overlay = document.createElement("button");
        overlay.className = "autoplay-overlay";
        overlay.textContent = "\u25B6 Play";
        overlay.addEventListener("click", function () {
          audioPlayer.play();
          overlay.remove();
        });
        card.appendChild(overlay);
      });
    }
  }

  fetch(`/api/document/${documentId}`)
    .then((r) => {
      if (!r.ok) throw new Error("Failed to load transcript");
      return r.json();
    })
    .then((data) => {
      container.innerHTML = "";
      buildTranscript(data);

      if (seekTime > 0) {
        // Scroll the matching alignment into view
        const target = alignmentMap.find(
          (a) => seekTime >= a.start && seekTime < a.end
        );
        if (target) {
          target.el.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }
    })
    .catch(() => {
      container.innerHTML =
        '<div class="transcript-loading">Failed to load transcript.</div>';
    });

  function buildTranscript(data) {
    if (!data.speeches) return;

    // Build regex for search term highlighting
    const searchRe = searchQuery
      ? new RegExp(
          `(${searchQuery.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`,
          "gi"
        )
      : null;

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
          wordSpan.dataset.start = word.start;
          wordSpan.dataset.end = word.end;

          if (searchRe) {
            wordSpan.innerHTML = word.text.replace(
              searchRe,
              '<span class="search-match">$1</span>'
            );
          } else {
            wordSpan.textContent = word.text;
          }

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
  }

  /* --- Highlighting --- */

  function binarySearch(segments, t) {
    let lo = 0;
    let hi = segments.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (t < segments[mid].start) hi = mid - 1;
      else if (t >= segments[mid].end) lo = mid + 1;
      else return segments[mid];
    }
    return null;
  }

  function updateHighlight() {
    const t = audioPlayer.currentTime;

    const curWord = binarySearch(wordMap, t);
    if (curWord && curWord.el !== prevWord) {
      if (prevWord) prevWord.classList.remove("highlight-word");
      curWord.el.classList.add("highlight-word");
      prevWord = curWord.el;
    }

    const curAlignment = binarySearch(alignmentMap, t);
    if (curAlignment && curAlignment.el !== prevAlignment) {
      if (prevAlignment) prevAlignment.classList.remove("highlight-sentence");
      curAlignment.el.classList.add("highlight-sentence");
      prevAlignment = curAlignment.el;

      // Auto-scroll to keep active sentence visible
      curAlignment.el.scrollIntoView({ behavior: "smooth", block: "nearest" });
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
})();

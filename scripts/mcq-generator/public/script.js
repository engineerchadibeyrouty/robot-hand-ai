// State
let selectedFile = null;
let questions = [];
let currentQ = 0;
let score = 0;
let userAnswers = [];
let answered = false;

// DOM elements
const uploadScreen = document.getElementById('uploadScreen');
const loadingScreen = document.getElementById('loadingScreen');
const errorScreen = document.getElementById('errorScreen');
const quizScreen = document.getElementById('quizScreen');
const resultsScreen = document.getElementById('resultsScreen');
const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const fileBadge = document.getElementById('fileBadge');
const fileNameEl = document.getElementById('fileName');
const removeFileBtn = document.getElementById('removeFile');
const generateBtn = document.getElementById('generateBtn');
const loadingMsg = document.getElementById('loadingMsg');

// File handling
fileInput.addEventListener('change', e => handleFile(e.target.files[0]));

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  handleFile(e.dataTransfer.files[0]);
});

removeFileBtn.addEventListener('click', e => {
  e.stopPropagation();
  clearFile();
});

function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  const name = file.name.length > 35 ? file.name.substring(0, 32) + '...' : file.name;
  fileNameEl.textContent = name;
  fileBadge.classList.remove('hidden');
  generateBtn.disabled = false;
}

function clearFile() {
  selectedFile = null;
  fileInput.value = '';
  fileBadge.classList.add('hidden');
  generateBtn.disabled = true;
}

// Screen switching
function showScreen(name) {
  [uploadScreen, loadingScreen, errorScreen, quizScreen, resultsScreen]
    .forEach(s => s.classList.add('hidden'));
  const screens = {
    upload: uploadScreen,
    loading: loadingScreen,
    error: errorScreen,
    quiz: quizScreen,
    results: resultsScreen
  };
  screens[name].classList.remove('hidden');
}

function showError(msg) {
  document.getElementById('errorText').textContent = msg;
  showScreen('error');
}

// Generate quiz
generateBtn.addEventListener('click', generateQuiz);

async function generateQuiz() {
  if (!selectedFile) return;

  const numQuestions = parseInt(document.getElementById('numQuestions').value) || 10;
  const difficulty = document.getElementById('difficulty').value;

  showScreen('loading');

  const msgs = [
    'Reading your document...',
    'Analyzing key concepts...',
    'Crafting questions...',
    'Almost there...'
  ];
  let mi = 0;
  const interval = setInterval(() => {
    mi++;
    if (mi < msgs.length) loadingMsg.textContent = msgs[mi];
  }, 3000);

  try {
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('numQuestions', numQuestions);
    formData.append('difficulty', difficulty);

    const resp = await fetch('/api/generate', {
      method: 'POST',
      body: formData
    });

    clearInterval(interval);

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.error || 'Server error');
    }

    const data = await resp.json();
    questions = data.questions;

    if (!Array.isArray(questions) || questions.length === 0) {
      throw new Error('No questions were generated');
    }

    currentQ = 0;
    score = 0;
    userAnswers = [];
    answered = false;
    renderQuestion();
    showScreen('quiz');

  } catch (err) {
    clearInterval(interval);
    console.error(err);
    showError('Failed to generate quiz. ' + (err.message || 'Please try again.'));
  }
}

// Quiz logic
function renderQuestion() {
  const q = questions[currentQ];
  document.getElementById('qCounter').textContent = `${currentQ + 1} / ${questions.length}`;
  document.getElementById('scoreLive').textContent = `${score} correct`;
  document.getElementById('progressFill').style.width = `${(currentQ / questions.length) * 100}%`;
  document.getElementById('qText').textContent = q.question;

  const container = document.getElementById('optionsContainer');
  container.innerHTML = '';
  const letters = ['A', 'B', 'C', 'D'];

  q.options.forEach((opt, i) => {
    const btn = document.createElement('button');
    btn.className = 'option-btn';
    const label = opt.replace(/^[A-D]\)\s*/, '');
    btn.innerHTML = `<span class="letter">${letters[i]}</span><span>${label}</span>`;
    btn.addEventListener('click', () => selectAnswer(i));
    container.appendChild(btn);
  });

  document.getElementById('explanationBox').classList.add('hidden');
  document.getElementById('nextBtn').classList.add('hidden');
  answered = false;
}

function selectAnswer(idx) {
  if (answered) return;
  answered = true;

  const q = questions[currentQ];
  const correct = q.correct;
  userAnswers.push(idx);

  const btns = document.querySelectorAll('.option-btn');
  btns.forEach((b, i) => {
    b.disabled = true;
    if (i === correct) b.classList.add('correct');
    if (i === idx && idx !== correct) b.classList.add('wrong');
  });

  if (idx === correct) score++;
  document.getElementById('scoreLive').textContent = `${score} correct`;

  const expBox = document.getElementById('explanationBox');
  expBox.innerHTML = `<strong>${idx === correct ? '✓ Correct!' : '✗ Incorrect.'}</strong> ${q.explanation || ''}`;
  expBox.classList.remove('hidden');

  const nextBtn = document.getElementById('nextBtn');
  nextBtn.textContent = currentQ < questions.length - 1 ? 'Next Question' : 'See Results';
  nextBtn.classList.remove('hidden');
}

document.getElementById('nextBtn').addEventListener('click', () => {
  currentQ++;
  if (currentQ < questions.length) {
    renderQuestion();
  } else {
    showResults();
  }
});

// Results
function showResults() {
  showScreen('results');
  const pct = Math.round((score / questions.length) * 100);
  document.getElementById('scorePct').textContent = pct + '%';
  document.getElementById('scoreDetail').textContent = `${score} / ${questions.length}`;

  const circumference = 2 * Math.PI * 60;
  const offset = circumference - (pct / 100) * circumference;
  setTimeout(() => {
    document.getElementById('ringFill').style.strokeDashoffset = offset;
  }, 100);

  let title, subtitle;
  if (pct >= 90) { title = 'Excellent!'; subtitle = 'You really know this material.'; }
  else if (pct >= 70) { title = 'Good job!'; subtitle = 'Solid understanding overall.'; }
  else if (pct >= 50) { title = 'Not bad!'; subtitle = 'A few areas to review.'; }
  else { title = 'Keep studying!'; subtitle = 'Review the material and try again.'; }

  document.getElementById('resultTitle').textContent = title;
  document.getElementById('resultSubtitle').textContent = subtitle;

  // Build review list
  const reviewList = document.getElementById('reviewList');
  reviewList.innerHTML = '';
  const letters = ['A', 'B', 'C', 'D'];

  questions.forEach((q, i) => {
    const div = document.createElement('div');
    div.className = 'review-item';
    const userAns = userAnswers[i];
    const isCorrect = userAns === q.correct;
    const cleanOpt = o => o.replace(/^[A-D]\)\s*/, '');

    div.innerHTML = `
      <div class="q-label">Question ${i + 1}</div>
      <div class="q-text">${q.question}</div>
      ${!isCorrect ? `<div class="answer-line your-wrong">Your answer: ${letters[userAns]}) ${cleanOpt(q.options[userAns])}</div>` : ''}
      <div class="answer-line correct-ans">${isCorrect ? '✓' : '→'} Correct: ${letters[q.correct]}) ${cleanOpt(q.options[q.correct])}</div>
      <div class="r-explanation">${q.explanation || ''}</div>
    `;
    reviewList.appendChild(div);
  });
}

document.getElementById('toggleReview').addEventListener('click', () => {
  const reviewList = document.getElementById('reviewList');
  reviewList.classList.toggle('hidden');
  document.getElementById('toggleReview').textContent =
    reviewList.classList.contains('hidden') ? 'Review Answers' : 'Hide Review';
});

function retakeQuiz() {
  currentQ = 0;
  score = 0;
  userAnswers = [];
  answered = false;
  renderQuestion();
  showScreen('quiz');
}

function resetAll() {
  clearFile();
  questions = [];
  currentQ = 0;
  score = 0;
  userAnswers = [];
  showScreen('upload');
}
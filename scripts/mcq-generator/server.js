require('dotenv').config();
const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const pdfParse = require('pdf-parse');
const Anthropic = require('@anthropic-ai/sdk').default;

const app = express();
const upload = multer({ dest: 'uploads/' });
const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// Serve the frontend
app.use(express.static('public'));

// Upload and generate quiz
app.post('/api/generate', upload.single('file'), async (req, res) => {
  try {
    const file = req.file;
    if (!file) return res.status(400).json({ error: 'No file uploaded' });

    const numQuestions = parseInt(req.body.numQuestions) || 10;
    const difficulty = req.body.difficulty || 'medium';

    // Extract text from file
    let text = '';
    const ext = path.extname(file.originalname).toLowerCase();

    if (ext === '.pdf') {
      const buffer = fs.readFileSync(file.path);
      const data = await pdfParse(buffer);
      text = data.text;
    } else {
      text = fs.readFileSync(file.path, 'utf-8');
    }

    // Clean up uploaded file
    fs.unlinkSync(file.path);

    // Trim to 50k characters max
    text = text.substring(0, 200000);

    if (text.trim().length < 50) {
      return res.status(400).json({ error: 'File has too little text to generate questions.' });
    }

    // Call Claude API
    const message = await client.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 4096,
      messages: [{
        role: 'user',
        content: `You are a quiz generator. Given the following study material, generate exactly ${numQuestions} multiple-choice questions at ${difficulty} difficulty.

Rules:
- Each question must have exactly 4 options (A, B, C, D)
- Only one correct answer per question
- Questions should cover different parts of the material
- ${difficulty === 'easy' ? 'Focus on definitions, key terms, and direct recall.' : difficulty === 'medium' ? 'Focus on understanding concepts and applying knowledge.' : 'Focus on analysis, comparison, edge cases, and deeper application.'}
- Include a brief explanation for why the correct answer is right
- Respond ONLY with valid JSON, no markdown fences, no preamble

JSON format:
[{"question":"...","options":["A) ...","B) ...","C) ...","D) ..."],"correct":0,"explanation":"..."}]

Where "correct" is the 0-based index of the correct option.

STUDY MATERIAL:
${text}`
      }]
    });

    const responseText = message.content
      .map(block => block.text || '')
      .join('');
    const clean = responseText.replace(/```json|```/g, '').trim();
    const questions = JSON.parse(clean);

    res.json({ questions });

  } catch (err) {
    console.error('Error:', err.message);
    res.status(500).json({ error: 'Failed to generate quiz. ' + err.message });
  }
});

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
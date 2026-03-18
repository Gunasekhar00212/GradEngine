
# 🚀 GradEngine

GradEngine is an AI-based exam evaluation prototype that assists in checking handwritten student answer sheets using a rubric-based scoring approach.

It is designed to simulate how a teacher evaluates answers — focusing on **concept presence** rather than exact wording.

---

## 🎯 Problem Statement

Manual evaluation of exam papers is:

- Time-consuming
- Inconsistent across evaluators
- Difficult to scale for large batches

GradEngine aims to:

- Reduce evaluation workload
- Provide consistent scoring support
- Assist teachers with semi-automated grading

---

## 🧠 Core Idea

Instead of matching exact answers, GradEngine:

1. Extracts text from handwritten scripts
2. Converts answers into semantic representations
3. Compares them with rubric concepts
4. Assigns marks based on concept similarity

---

## ⚙️ Current Prototype Scope

This is a **proof-of-concept prototype**, not a full system.

### Current limitations:

- Supports only **one student PDF**
- Assumes **single question**
- Uses **hardcoded rubric**
- No database or multi-user support yet

---

## 🏗️ System Architecture

```

PDF (Student Answer Sheet)
↓
Convert PDF → Images
↓
Gemini API (Handwriting → Text)
↓
Text Storage (JSON - optional)
↓
Sentence-BERT (Text → Vectors)
↓
Cosine Similarity (Concept Matching)
↓
Rubric-Based Scoring
↓
Final Score Output

```

---

## 🔧 Tech Stack

### 🔹 Core Technologies

- **Python** – Main programming language
- **Google Gemini API (google-genai)** – Handwritten text extraction
- **Sentence-Transformers (BERT)** – Semantic embeddings
- **Scikit-learn** – Similarity computation

### 🔹 Supporting Tools

- **pdf2image** – Convert PDF to images
- **OpenCV / PIL** – Image handling (optional)
- **dotenv** – API key management

---

## 📂 Project Structure

```

GradEngine/
│
├── main.py
│
├── processing/
│   ├── pdf_split.py
│   ├── extract_text.py
│
├── scoring/
│   ├── embedding.py
│   ├── scoring_engine.py
│
├── data/
│   ├── pdfs/
│   ├── pages/
│   ├── extracted_text/
│
├── .env
├── .gitignore
└── README.md

```

---

## 🔄 How It Works (Step-by-Step)

### 1. Input

- A student answer sheet (PDF)

---

### 2. PDF → Images

- Each page is converted into an image

---

### 3. Handwriting Recognition

- Gemini API extracts readable text from images

---

### 4. Text Processing

- Extracted text is combined into a single answer

---

### 5. Embedding Generation

- Text is converted into vector representation using Sentence-BERT

---

### 6. Rubric Matching

- Each rubric concept is also converted into a vector
- Cosine similarity is used to detect concept presence

---

### 7. Scoring

- Marks are assigned based on detected concepts

---

### 8. Output

- Final score
- Detected concepts

---

## 📊 Example

### Rubric:

```

sunlight → 1
CO2 → 1
water → 1
oxygen → 1
glucose → 1

```

### Student Answer:

```

Plants use sunlight, CO2 and water to produce glucose and oxygen

```

### Output:

```

Detected Concepts: sunlight, CO2, water, oxygen, glucose
Final Score: 5 / 5

```

---

## ⚠️ Known Limitations

- Handwriting extraction depends on image quality
- Semantic similarity does not guarantee factual correctness
- Does not handle diagrams or complex math yet
- Not optimized for large-scale deployment

---

## 🚧 Future Improvements

- Multi-question support
- Multi-student batch processing
- Database integration
- Better handwriting handling (custom dataset)
- Diagram and math evaluation support
- Frontend dashboard for teachers



---

## 📌 Note

This project is a **learning and research prototype**, aimed at exploring AI-assisted evaluation systems — not a production-ready grading system.

---

```

---


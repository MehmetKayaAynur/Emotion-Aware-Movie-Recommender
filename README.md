# Emotion-Aware Movie Recommender

This project recommends movies based on the user's emotional state.
The system analyzes user mood from text input, extracts emotional features from movie reviews,
and matches them using an emotion-based similarity approach.

The project is developed in a **modular architecture** with three main components.

---

## Project Modules

### Module 1 – Emotion / Mood Analysis (Text → Emotion)
- Takes user input text (Turkish or English)
- Translates Turkish input to English
- Predicts emotions using a pre-trained GoEmotions model
- Outputs:
  - Full emotion probability distribution
  - Active emotions above a defined threshold

### Module 2 – Movie Database & Emotion Feature Extraction
- Processes a large movie dataset (≈ 4800 movies)
- Analyzes movie reviews using the same emotion model
- Removes the `neutral` emotion for stronger semantic signals
- Generates:
  - Emotion vectors per movie
  - Dominant emotions for each movie
- Stores results in a structured CSV database

### Module 3 – Recommendation Engine & User Interface
- Streamlit-based interactive UI
- Accepts user mood text input
- Matches user emotion vector with movie emotion vectors
- Ranks and recommends the most emotionally compatible movies
- Fetches and displays movie posters using TMDb API

---

##  Technologies Used
- Python 3
- HuggingFace Transformers
- RoBERTa (GoEmotions)
- PyTorch
- Streamlit
- Pandas / NumPy
- TMDb API (for movie posters)

---

##  TMDb API Key Setup (Required for Posters)

Movie posters are fetched via TMDb API.

### Step 1: Create a `.env` file
```bash
cp .env.example .env

# 🎮 Steam-200k Collaborative Filtering Recommender

This project implements two classic **Collaborative Filtering (CF)** algorithms for personalized game recommendation using the **Steam-200k dataset**.  
Both algorithms are designed to analyze user–game interaction data (purchase and play records) and recommend games a user might enjoy.

---

## 📘 Overview

### 1. User-Based Collaborative Filtering (UserCF)

**Idea:** Find users with similar gaming preferences and recommend games they liked.  

**Steps:**
1. Build a user–game interaction matrix.  
2. Compute similarity between users (cosine similarity based on co-played games).  
3. For a target user, find top-K most similar users.  
4. Recommend games those users played but the target user hasn’t.  

---

### 2. Item-Based Collaborative Filtering (ItemCF)

**Idea:** Recommend games similar to those the user has already played.  

**Steps:**
1. Build a game–user matrix.  
2. Compute similarity between games (cosine similarity based on shared players).  
3. For each game the user has played, find its top-K similar games.  
4. Aggregate and rank candidate games by similarity-weighted scores.  

---

## ⚙️ Key Features

- Implements both **UserCF** and **ItemCF** with clear modular design.  
- Uses **cosine similarity** over implicit feedback (purchase/play events).  
- Simple **GUI interface** via `easygui` for interactive recommendation testing.  
- Compatible with the **Steam-200k dataset** (CSV format).  

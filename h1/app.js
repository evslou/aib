// app.js (ES module version using transformers.js for local sentiment classification)

import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6/dist/transformers.min.js";

// Global variables
let reviews = [];
let apiToken = ""; // kept for UI compatibility, but not used with local inference
let sentimentPipeline = null; // transformers.js text-classification pipeline

// Hardcoded Google Sheets Web App URL (replace with your own)
const GAS_WEB_APP_URL = "https://script.google.com/macros/s/AKfycbxt0k1VUS7bqy0HMJtNttYh3YvUwMGCHKOhnTR6XTiG45LCjz7bjXRc-en7ixZDtS_MMg/exec";

// DOM elements
const analyzeBtn = document.getElementById("analyze-btn");
const reviewText = document.getElementById("review-text");
const sentimentResult = document.getElementById("sentiment-result");
const loadingElement = document.querySelector(".loading");
const errorElement = document.getElementById("error-message");
const apiTokenInput = document.getElementById("api-token");
const statusElement = document.getElementById("status"); // optional status label for model loading

// Initialize the app
document.addEventListener("DOMContentLoaded", function () {
  // Load the TSV file (Papa Parse)
  loadReviews();

  // Set up event listeners
  analyzeBtn.addEventListener("click", analyzeRandomReview);
  apiTokenInput.addEventListener("change", saveApiToken);

  // Load saved API token if exists (not used with local inference but kept for UI)
  const savedToken = localStorage.getItem("hfApiToken");
  if (savedToken) {
    apiTokenInput.value = savedToken;
    apiToken = savedToken;
  }

  // Initialize transformers.js sentiment model
  initSentimentModel();
});

/**
 * Determines the appropriate business action based on sentiment analysis results.
 * @param {number} confidence - Confidence score (0.0 to 1.0)
 * @param {string} label - "POSITIVE" or "NEGATIVE"
 * @returns {object} { actionCode, uiMessage, uiColor }
 */
function determineBusinessAction(confidence, label) {
    // Normalize score to a 0 (worst) - 1 (best) scale
    let normalizedScore = 0.5; // default neutral
    const upperLabel = label ? label.toUpperCase() : '';

    if (upperLabel === "POSITIVE") {
        normalizedScore = confidence; // e.g., 0.9 -> 0.9 (great)
    } else if (upperLabel === "NEGATIVE") {
        normalizedScore = 1.0 - confidence; // e.g., 0.9 conf -> 0.1 (terrible)
    }

    // Apply business thresholds
    if (normalizedScore <= 0.4) {
        return {
            actionCode: "OFFER_COUPON",
            uiMessage: "🚨 We are truly sorry. Please accept this 50% discount coupon.",
            uiColor: "#ef4444" // red
        };
    } else if (normalizedScore < 0.7) {
        return {
            actionCode: "REQUEST_FEEDBACK",
            uiMessage: "📝 Thank you! Could you tell us how we can improve?",
            uiColor: "#6b7280" // gray
        };
    } else {
        return {
            actionCode: "ASK_REFERRAL",
            uiMessage: "⭐ Glad you liked it! Refer a friend and earn rewards.",
            uiColor: "#3b82f6" // blue
        };
    }
}


// Initialize transformers.js text-classification pipeline with a supported model
async function initSentimentModel() {
  try {
    if (statusElement) {
      statusElement.textContent = "Loading sentiment model...";
    }

    // Use a transformers.js-supported text-classification model.
    // Xenova/distilbert-base-uncased-finetuned-sst-2-english is a common choice.
    sentimentPipeline = await pipeline(
      "text-classification",
      "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
    );

    if (statusElement) {
      statusElement.textContent = "Sentiment model ready";
    }
  } catch (error) {
    console.error("Failed to load sentiment model:", error);
    showError(
      "Failed to load sentiment model. Please check your network connection and try again."
    );
    if (statusElement) {
      statusElement.textContent = "Model load failed";
    }
  }
}

// Load and parse the TSV file using Papa Parse
function loadReviews() {
  fetch("reviews_test.tsv")
    .then((response) => {
      if (!response.ok) {
        throw new Error("Failed to load TSV file");
      }
      return response.text();
    })
    .then((tsvData) => {
      Papa.parse(tsvData, {
        header: true,
        delimiter: "\t",
        complete: (results) => {
          reviews = results.data
            .map((row) => row.text)
            .filter((text) => typeof text === "string" && text.trim() !== "");
          console.log("Loaded", reviews.length, "reviews");
        },
        error: (error) => {
          console.error("TSV parse error:", error);
          showError("Failed to parse TSV file: " + error.message);
        },
      });
    })
    .catch((error) => {
      console.error("TSV load error:", error);
      showError("Failed to load TSV file: " + error.message);
    });
}

// Save API token to localStorage (UI compatibility; not used with local inference)
function saveApiToken() {
  apiToken = apiTokenInput.value.trim();
  if (apiToken) {
    localStorage.setItem("hfApiToken", apiToken);
  } else {
    localStorage.removeItem("hfApiToken");
  }
}

// Analyze a random review
function analyzeRandomReview() {
  hideError();

  if (!Array.isArray(reviews) || reviews.length === 0) {
    showError("No reviews available. Please try again later.");
    return;
  }

  if (!sentimentPipeline) {
    showError("Sentiment model is not ready yet. Please wait a moment.");
    return;
  }

  const selectedReview =
    reviews[Math.floor(Math.random() * reviews.length)];

  // Display the review
  reviewText.textContent = selectedReview;

  // Show loading state
  loadingElement.style.display = "block";
  analyzeBtn.disabled = true;
  sentimentResult.innerHTML = ""; // Reset previous result
  sentimentResult.className = "sentiment-result"; // Reset classes

  // Call local sentiment model (transformers.js)
  analyzeSentiment(selectedReview)
    .then((result) => displaySentiment(result))
    .catch((error) => {
      console.error("Error:", error);
      showError(error.message || "Failed to analyze sentiment.");
    })
    .finally(() => {
      loadingElement.style.display = "none";
      analyzeBtn.disabled = false;
    });
}

// Call local transformers.js pipeline for sentiment classification
async function analyzeSentiment(text) {
  if (!sentimentPipeline) {
    throw new Error("Sentiment model is not initialized.");
  }

  // transformers.js text-classification pipeline returns:
  // [{ label: 'POSITIVE', score: 0.99 }, ...]
  const output = await sentimentPipeline(text);

  if (!Array.isArray(output) || output.length === 0) {
    throw new Error("Invalid sentiment output from local model.");
  }

  // Wrap to match [[{ label, score }]] shape expected by displaySentiment
  return [output];
}
// Function to log to Google Sheets (now uses hardcoded URL)
async function logToGoogleSheets(review, sentimentStr, actionCode, meta) {
  const formData = new URLSearchParams();
  formData.append('review', review);
  formData.append('sentiment', sentimentStr);
  formData.append('action', actionCode);      // new parameter
  formData.append('meta', JSON.stringify(meta));

  try {
    const response = await fetch(GAS_WEB_APP_URL, {
      method: 'POST',
      body: formData
    });
    if (!response.ok) {
      console.warn('Failed to log to Google Sheets:', response.statusText);
    }
  } catch (error) {
    console.warn('Error logging to Google Sheets:', error);
  }
}

// Display sentiment result
function displaySentiment(result) {
  // Default to neutral if we can't parse the result
  let sentiment = "neutral";
  let score = 0.5;
  let label = "NEUTRAL";

  // Expected format: [[{label: 'POSITIVE', score: 0.99}]]
  if (
    Array.isArray(result) &&
    result.length > 0 &&
    Array.isArray(result[0]) &&
    result[0].length > 0
  ) {
    const sentimentData = result[0][0];

    if (sentimentData && typeof sentimentData === "object") {
      label =
        typeof sentimentData.label === "string"
          ? sentimentData.label.toUpperCase()
          : "NEUTRAL";
      score =
        typeof sentimentData.score === "number"
          ? sentimentData.score
          : 0.5;

      // Determine sentiment bucket
      if (label === "POSITIVE" && score > 0.5) {
        sentiment = "positive";
      } else if (label === "NEGATIVE" && score > 0.5) {
        sentiment = "negative";
      } else {
        sentiment = "neutral";
      }
    }
  }
  // Build sentiment string (e.g., "POSITIVE (95.0%)")
  const sentimentStr = `${label} (${(score * 100).toFixed(1)}%)`;

  // Update UI (as before)
  sentimentResult.classList.add(sentiment);
  sentimentResult.innerHTML = `
        <i class="fas ${getSentimentIcon(sentiment)} icon"></i>
        <span>${sentimentStr}</span>
    `;
  // --- NEW: Business Action Logic ---
  const decision = determineBusinessAction(score, label);
  
  // Display the action message in the new div
  const actionDiv = document.getElementById('action-result');
  actionDiv.textContent = decision.uiMessage;
  actionDiv.style.backgroundColor = decision.uiColor + '20'; // 20 = 12% opacity for background
  actionDiv.style.color = decision.uiColor;
  actionDiv.style.border = `2px solid ${decision.uiColor}`;
  actionDiv.classList.add('show'); // make it visible

  // --- Logging to Google Sheets ---
  const meta = {
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    language: navigator.language,
    screen: `${screen.width}x${screen.height}`,
    timestamp: new Date().toISOString()
  };

  const review = reviewText.textContent;
  
  // Fire-and-forget log with action code
  logToGoogleSheets(review, sentimentStr, decision.actionCode, meta);
}

// Get appropriate icon for sentiment bucket
function getSentimentIcon(sentiment) {
  switch (sentiment) {
    case "positive":
      return "fa-thumbs-up";
    case "negative":
      return "fa-thumbs-down";
    default:
      return "fa-question-circle";
  }
}

// Show error message
function showError(message) {
  errorElement.textContent = message;
  errorElement.style.display = "block";
}

// Hide error message
function hideError() {
  errorElement.style.display = "none";
}

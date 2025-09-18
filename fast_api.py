from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import os
from pathlib import Path
from contextlib import asynccontextmanager
from utils import product_tokenizer

# Define the product_tokenizer function here (same as in notebook)
def product_tokenizer(text):
    """Tokenize product strings by splitting on spaces"""
    return text.split()

# Global variable to store model data
model_data = None


class CustomerInput(BaseModel):
    """Simplified input model for customer data"""
    city: Optional[str] = None
    total_spend: Optional[float] = None
    items_purchased: Optional[int] = None
    average_rating: Optional[float] = None
    discount_applied: Optional[bool] = None
    days_since_last_purchase: Optional[int] = None
    satisfaction_level: Optional[str] = "Neutral"

    # Optional extras
    gender: Optional[str] = None
    age: Optional[int] = None
    membership_type: Optional[str] = None


def recommend_products_api(customer: CustomerInput, top_k: int = 5) -> List[Dict[str, float]]:
    """Generate recommendations using the trained model"""
    if model_data is None:
        return get_fallback_recommendations(customer)

    try:
        # Map customer inputs to the expected format with better defaults
        customer_dict = {
            'Gender': str(customer.gender).capitalize() if customer.gender else "Male",
            'Age': int(customer.age) if customer.age is not None else 30,
            'City': str(customer.city) if customer.city else "New York",
            'Membership Type': str(customer.membership_type).capitalize() if customer.membership_type else "Bronze",
            'Total Spend': float(customer.total_spend) if customer.total_spend is not None else 100.0,
            'Items Purchased': int(customer.items_purchased) if customer.items_purchased is not None else 5,
            'Average Rating': float(customer.average_rating) if customer.average_rating is not None else 3.5,
            'Discount Applied': bool(customer.discount_applied) if customer.discount_applied is not None else False,
            'Days Since Last Purchase': int(customer.days_since_last_purchase) if customer.days_since_last_purchase is not None else 30,
            'Satisfaction Level': str(customer.satisfaction_level).capitalize() if customer.satisfaction_level else "Neutral",
            'Assigned Products': ""
        }

        print("Customer Dict:", customer_dict)

        # Convert to DataFrame
        customer_df = pd.DataFrame([customer_dict])

        # Apply preprocessing
        X_processed = model_data['preprocessor'].transform(customer_df)
        X_arr = X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=1.0, neginf=0.0)

        # Feature names
        cat_names = model_data['preprocessor'].named_transformers_['cat'].get_feature_names_out(
            ['Gender', 'City', 'Membership Type', 'Satisfaction Level']
        )
        prod_names = model_data['preprocessor'].named_transformers_['products'].get_feature_names_out()
        num_names = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating',
                     'Discount Applied', 'Days Since Last Purchase']
        all_names = list(cat_names) + list(prod_names) + list(num_names)

        feature_df = pd.DataFrame(X_arr, columns=all_names)
        X_final = feature_df[model_data['feature_columns']].fillna(0).values

        # Model predictions
        predictions = model_data['model'].predict_proba(X_final)

        # Handle MultiOutputClassifier predictions correctly
        if isinstance(predictions, list):
            # MultiOutputClassifier returns list of arrays
            probs = []
            for pred in predictions:
                if pred.ndim > 1 and pred.shape[1] > 1:
                    # Binary classifier - take positive class probability
                    probs.append(pred[0, 1])
                else:
                    probs.append(pred[0])
            probs = np.array(probs, dtype=float)
        else:
            probs = np.array(predictions).flatten()

        # Clean probabilities
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

        # Add some randomization to break ties and create variation
        # Add small random noise to probabilities (±5%)
        noise = np.random.normal(0, 0.05, len(probs))
        probs = probs + noise
        probs = np.clip(probs, 0, 1)  # Keep probabilities in valid range

        print("Probs after noise:", probs[:10])  # Debug first 10 values

        # Sort & pick top-k
        top_k_indices = np.argsort(probs)[-top_k:][::-1]
        recommended_products = model_data['mlb'].classes_[top_k_indices]
        recommendation_scores = probs[top_k_indices]

        return [
            {"product": product, "score": round(float(score), 3)}
            for product, score in zip(recommended_products, recommendation_scores)
        ]

    except Exception as e:
        print(f"Error in model prediction: {e}")
        return get_fallback_recommendations(customer)


class RecommendationItem(BaseModel):
    product: str
    score: float


class RecommendationResponse(BaseModel):
    customer_profile: Dict
    recommendations: List[RecommendationItem]
    status: str


import pickle

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Patch: if pickle looks in __main__ for product_tokenizer, redirect to utils
        if name == "product_tokenizer":
            import utils
            return utils.product_tokenizer
        return super().find_class(module, name)


def load_model():
    """Load the trained model with custom unpickler to fix product_tokenizer reference"""
    global model_data
    try:
        current_file_dir = Path(__file__).resolve().parent
        model_path = current_file_dir / "models" / "recommendation_model.pkl"

        print(f"Looking for model at: {model_path}")
        print(f"File exists: {model_path.exists()}")

        if not model_path.exists():
            print(f"Current working directory: {Path.cwd()}")
            print(f"FastAPI file directory: {current_file_dir}")
            print(f"Contents of FastAPI directory: {list(current_file_dir.iterdir())}")
            return False

        with open(model_path, "rb") as f:
            model_data = CustomUnpickler(f).load()

        print("Model loaded successfully!")
        return True

    except FileNotFoundError:
        print("Model file not found. Please ensure the model is trained and saved.")
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


# ---------------- Lifespan handler (replaces @app.on_event("startup")) ---------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    success = load_model()
    if not success:
        print("Warning: Model could not be loaded. API will use fallback recommendations.")
    yield  # app runs here
    # Cleanup code (if needed) runs after app shutdown


# Initialize FastAPI with lifespan
app = FastAPI(
    title="E-commerce Recommendation API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {"message": "E-commerce Recommendation API", "status": "active", "model_loaded": model_data is not None}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_data is not None}


def get_fallback_recommendations(customer: CustomerInput) -> List[Dict[str, float]]:
    """Enhanced rule-based fallback recommendations with more variation"""
    recommendations = []

    age = customer.age if customer.age is not None else 30
    membership = customer.membership_type.lower() if customer.membership_type else "basic"
    spend = customer.total_spend if customer.total_spend is not None else 100
    city = customer.city.lower() if customer.city else "unknown"

    # Create more varied recommendations based on multiple factors
    all_products = {
        'Electronics': ['Smartphone', 'Laptop', 'Headphones', 'Tablet', 'Camera'],
        'Fashion': ['T-Shirt', 'Jeans', 'Sneakers', 'Jacket', 'Sunglasses'], 
        'Home': ['Blender', 'Vacuum', 'Lamp', 'Microwave', 'Fan'],
        'Sports': ['Football', 'TennisRacket', 'RunningShoes', 'YogaMat', 'Dumbbells'],
        'Beauty': ['Lipstick', 'Perfume', 'Moisturizer', 'FaceWash', 'HairSerum']
    }

    # Logic based on customer profile
    if age < 25:
        category_weights = {'Electronics': 0.4, 'Fashion': 0.3, 'Sports': 0.2, 'Beauty': 0.1}
    elif age < 35:
        category_weights = {'Electronics': 0.3, 'Fashion': 0.2, 'Home': 0.2, 'Sports': 0.15, 'Beauty': 0.15}
    else:
        category_weights = {'Home': 0.3, 'Fashion': 0.2, 'Beauty': 0.2, 'Electronics': 0.15, 'Sports': 0.15}

    if spend > 500:
        category_weights = {k: v*1.2 if k in ['Electronics', 'Home'] else v*0.8 
                          for k, v in category_weights.items()}

    if membership == 'gold':
        category_weights = {k: v*1.1 if k in ['Electronics'] else v 
                          for k, v in category_weights.items()}

    # Select products based on weights
    selected_products = []
    for category, weight in category_weights.items():
        n_products = max(1, int(weight * 5))  # Select 1-5 products per category
        products = np.random.choice(all_products[category], 
                                  min(n_products, len(all_products[category])), 
                                  replace=False)
        for product in products:
            score = weight + np.random.normal(0, 0.1)  # Add some randomness
            selected_products.append({"product": product, "score": max(0.1, score)})

    # Sort by score and return top 5
    selected_products.sort(key=lambda x: x["score"], reverse=True)
    
    # Add some randomization to scores to create variation
    for item in selected_products[:5]:
        item["score"] = round(float(item["score"]) + np.random.uniform(-0.1, 0.1), 3)
    
    return selected_products[:5]


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(customer: CustomerInput, top_k: int = 5):
    try:
        if top_k < 1 or top_k > 20:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")

        if customer.age is not None:
            if customer.age < 0 or customer.age > 120:
                raise HTTPException(status_code=400, detail="Age must be between 0 and 120")

        recommendations = recommend_products_api(customer, top_k)

        customer_profile = {
            "age": customer.age,
            "membership_type": customer.membership_type,
            "total_spend": customer.total_spend,
            "satisfaction_level": customer.satisfaction_level,
        }

        return RecommendationResponse(customer_profile=customer_profile, recommendations=recommendations, status="success")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
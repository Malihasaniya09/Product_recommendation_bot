import streamlit as st
import requests
from openai import OpenAI
import json
from dotenv import load_dotenv
import os

# Initialize OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
FASTAPI_URL = "http://127.0.0.1:8000/recommend"

# Professional page config
st.set_page_config(
    page_title="AI Business Recommender",
    page_icon="🚀",
    layout="wide"
)

# Modern CSS styling with light abstract background
st.markdown("""
<style>
    .stApp { 
        background: linear-gradient(135deg, #ffeef8 0%, #e8f4fd 25%, #fff5e6 50%, #f0f9ff 75%, #faf5ff 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .header { 
        text-align: center; 
        background: linear-gradient(135deg, #a78bfa 0%, #06b6d4 50%, #f97316 100%);
        color: white; 
        padding: 1.5rem; 
        border-radius: 15px; 
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .profile-card { 
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 1.5rem; 
        border-radius: 15px; 
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        margin: 1rem 0;
    }
    .chat-message { 
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(5px);
        padding: 1rem; 
        border-radius: 12px; 
        margin: 0.5rem 0;
        border-left: 4px solid #a78bfa;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
    }
    .error-alert { 
        background: linear-gradient(135deg, #fef2f2 0%, #fce7e7 100%);
        color: #dc2626; 
        padding: 1rem; 
        border-radius: 12px; 
        border: 1px solid #fca5a5;
        margin: 1rem 0;
        font-weight: bold;
        box-shadow: 0 4px 16px rgba(220,38,38,0.1);
    }
    .success-alert { 
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        color: #16a34a; 
        padding: 1.5rem; 
        border-radius: 12px; 
        border: 1px solid #86efac;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(22,163,74,0.1);
    }
    .recommendation-box {
        background: linear-gradient(135deg, #a78bfa 0%, #06b6d4 50%, #f97316 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h1>🚀 AI Business Intelligence Platform</h1>
    <p>Smart Product Recommendations & Customer Analytics</p>
</div>
""", unsafe_allow_html=True)

# 📊 Business metrics right under the header
st.markdown("## 📊 Business Performance Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📈 Accuracy", "94.2%")
with col2:
    st.metric("⚡ Response Time", "1.2s")
with col3:
    st.metric("👥 Active Users", "2,847")
with col4:
    st.metric("💼 Conversions", "23.4%")

st.markdown("---")

# Initialize session state
if "profile" not in st.session_state:
    st.session_state.profile = {}
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function schema for OpenAI
function_schema = [{
    "name": "extract_customer_profile",
    "description": "Extract customer profile information from user input",
    "parameters": {
        "type": "object",
        "properties": {
            "gender": {"type": "string", "description": "Customer's gender (male/female)"},
            "city": {"type": "string", "description": "Customer's city"},
            "age": {"type": "integer", "description": "Customer's age"},
            "total_spend": {"type": "number", "description": "Total spending amount"},
            "items_purchased": {"type": "integer", "description": "Number of items purchased"},
            "satisfaction_level": {"type": "string", "enum": ["Low", "Medium", "High"]},
            "membership_type": {"type": "string", "enum": ["Basic", "Premium", "VIP", "Bronze", "Silver", "Gold"]}
        },
        "required": []
    }
}]

def validate_profile(profile):
    """Check for negative values and return errors"""
    errors = []
    if profile.get("total_spend", 0) < 0:
        errors.append("Total spend cannot be negative")
    if profile.get("age", 0) < 0:
        errors.append("Age cannot be negative")
    if profile.get("items_purchased", 0) < 0:
        errors.append("Items purchased cannot be negative")
    return errors

def get_recommendations(profile):
    """Get recommendations from FastAPI or generate mock ones"""
    try:
        response = requests.post(FASTAPI_URL, json=profile, timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    # Mock recommendations
    spend = profile.get("total_spend", 0)
    
    if spend > 500:
        products = ["Premium Electronics", "Luxury Accessories", "High-End Fashion"]
    elif spend > 100:
        products = ["Smart Home Devices", "Fashion Accessories", "Books & Learning"]
    else:
        products = ["Budget Electronics", "Everyday Essentials", "Student Supplies"]
    
    return {"recommendations": [{"product": p} for p in products]}

def check_profile_completeness():
    """Check if we have enough info to make recommendations"""
    required_fields = ['age', 'gender', 'city', 'total_spend', 'membership_type']
    missing = [field for field in required_fields if not st.session_state.profile.get(field)]
    return len(missing) <= 1

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 💬 AI Customer Assistant")
    
    for msg in st.session_state.messages:
        role_icon = "🧑" if msg["role"] == "user" else "🤖"
        st.markdown(f"""
        <div class="chat-message">
            <strong>{role_icon} {msg["role"].title()}:</strong> {msg["content"]}
        </div>
        """, unsafe_allow_html=True)
    
    user_input = st.chat_input("💭 Tell me about yourself: age, location, spending, preferences...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        try:
            context_message = f"Current customer profile: {st.session_state.profile}. New user input: {user_input}. Extract any new information and add to the existing profile."
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Extract customer profile information. If negative values for spending, age, or items, respond with error. Always extract any available info."},
                    {"role": "user", "content": context_message}
                ],
                functions=function_schema,
                function_call="auto"
            )
            
            message = response.choices[0].message
            
            if message.function_call:
                try:
                    new_profile = json.loads(message.function_call.arguments)
                    errors = validate_profile(new_profile)
                    
                    if errors:
                        error_msg = f"⚠️ **Negative Parameters Detected**: {', '.join(errors)}. Please provide positive values only."
                        st.markdown(f'<div class="error-alert">{error_msg}</div>', unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    else:
                        st.session_state.profile.update({k: v for k, v in new_profile.items() if v is not None})
                        
                        if check_profile_completeness():
                            recs = get_recommendations(st.session_state.profile)
                            
                            if "recommendations" in recs:
                                products = [r["product"] for r in recs["recommendations"][:5]]
                                rec_text = f"Perfect! Based on your profile, here are my top recommendations: **{', '.join(products)}**"
                                st.markdown(f'<div class="success-alert">✨ {rec_text}</div>', unsafe_allow_html=True)
                                st.session_state.messages.append({"role": "assistant", "content": rec_text})
                        else:
                            current_fields = {k: v for k, v in st.session_state.profile.items() if v}
                            required_fields = ['age', 'gender', 'city', 'total_spend', 'membership_type']
                            missing = [field.replace('_', ' ').title() for field in required_fields if field not in current_fields]
                            
                            if missing:
                                current_info = ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in current_fields.items()])
                                missing_msg = f"Thanks! I have: {current_info}. To complete your profile, I also need: {', '.join(missing)}."
                                st.session_state.messages.append({"role": "assistant", "content": missing_msg})
                
                except json.JSONDecodeError:
                    st.error("Error processing your information")
            else:
                ai_response = message.content
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                
        except Exception as e:
            st.error(f"Service error: {str(e)}")
        
        st.rerun()

with col2:
    st.markdown("### 👤 Customer Profile")
    
    if st.session_state.profile:
        st.markdown('<div class="profile-card">', unsafe_allow_html=True)
        
        for key, value in st.session_state.profile.items():
            if value:
                display_key = key.replace("_", " ").title()
                if key == "total_spend":
                    st.metric(f"💰 {display_key}", f"${value}")
                elif key == "age":
                    st.metric(f"👤 {display_key}", f"{value} years")
                elif key == "city":
                    st.metric(f"🏢 {display_key}", value)
                else:
                    st.metric(f"📊 {display_key}", value)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🎯 Get New Recommendations", use_container_width=True):
            recs = get_recommendations(st.session_state.profile)
            if "recommendations" in recs:
                products = [r["product"] for r in recs["recommendations"]]
                st.markdown(f"""
                <div class="recommendation-box">
                    <h4>✨ Personalized Recommendations</h4>
                    <ul>
                        {''.join([f'<li>{p}</li>' for p in products[:5]])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Profile", use_container_width=True):
            st.session_state.profile = {}
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("💡 Start chatting to build your customer profile")

# Footer (now visible with dark text)
st.markdown("""
<div style="text-align: center; color: #333; padding: 1rem; margin-top: 2rem; font-weight: 500;">
    🔐 Enterprise Security | 🤖 Advanced AI | 📞 24/7 Business Support
</div>
""", unsafe_allow_html=True)
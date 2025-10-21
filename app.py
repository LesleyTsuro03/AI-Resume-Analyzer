import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import tempfile
import base64
import re
from database import (
    get_session, close_session, User, JobDescription, Candidate, Skill, Education, WorkExperience,
    create_user, verify_user, get_all_users, reset_user_password, 
    deactivate_user, activate_user, get_user_stats, get_recent_candidates,
    store_cv_file, get_cv_by_resume_code, get_cv_by_phone_number,
    delete_job_analysis, check_duplicate_candidate
)
from nlp_processor import ResumeParser
from ml_model import ScoringModel
import bcrypt

# Page configuration
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .high-score { color: #00cc96; }
    .medium-score { color: #ffa15c; }
    .low-score { color: #ef553b; }
    .superadmin-badge { 
        background-color: #ff4b4b; 
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
    }
    .user-management-table {
        font-size: 0.9em;
    }
    .report-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .strength-item {
        color: #00cc96;
        margin: 0.5rem 0;
    }
    .improvement-item {
        color: #ffa15c;
        margin: 0.5rem 0;
    }
    .confidential-code {
        background-color: #1f77b4;
        color: white;
        padding: 8px 12px;
        border-radius: 8px;
        font-family: monospace;
        font-weight: bold;
    }
    .cv-viewer {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class ResumeAnalyzerApp:
    def __init__(self):
        self.resume_parser = ResumeParser()
        self.scoring_model = ScoringModel()
        
        # Try to load pre-trained model
        if os.path.exists('model/trained_model.pkl'):
            self.scoring_model.load_model('model/trained_model.pkl')
    
    def login_page(self):
        """User authentication page"""
        st.markdown('<div class="main-header">AI Resume Analyzer</div>', unsafe_allow_html=True)
        
        # Initialize session state
        if 'user' not in st.session_state:
            st.session_state.user = None
        
        # If not logged in, show login form
        if st.session_state.user is None:
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.subheader("🔐 Login to Continue")
                
                with st.form("login_form"):
                    username = st.text_input("Username", placeholder="Enter your username")
                    password = st.text_input("Password", type="password", placeholder="Enter your password")
                    login_button = st.form_submit_button("Login", type="primary")
                    
                    if login_button:
                        if username and password:
                            user = verify_user(username, password)
                            if user:
                                st.session_state.user = user
                                st.success(f"Welcome back, {user.username}!")
                                st.rerun()
                            else:
                                st.error("Invalid username or password. Please try again.")
                        else:
                            st.warning("Please enter both username and password")
                
            
            return False
        
        return True
    
    def dashboard_page(self):
        """Main dashboard page"""
        st.title("📊 Dashboard")
        
        # Welcome message with user role
        if st.session_state.user.is_superadmin:
            st.success(f"👑 Welcome Super Admin: {st.session_state.user.username}")
        else:
            st.success(f"👤 Welcome: {st.session_state.user.username}")
        
        # Get statistics based on user role
        if st.session_state.user.is_superadmin:
            total_jobs, total_candidates = get_user_stats()
            total_users = len(get_all_users(st.session_state.user.user_id))
        else:
            total_jobs, total_candidates = get_user_stats(st.session_state.user.user_id)
            total_users = 1
        
        # Get average score (simplified for demo)
        avg_score = 75.5  # In real implementation, calculate from database
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Jobs", total_jobs)
        col2.metric("Total Candidates", total_candidates)
        col3.metric("Average Score", f"{avg_score:.1f}%")
        col4.metric("Active Users" if st.session_state.user.is_superadmin else "Your Account", 
                   total_users, 
                   "Super Admin" if st.session_state.user.is_superadmin else "Active")
        
        # Recent activity
        st.subheader("Recent Activity")
        recent_candidates = get_recent_candidates(
            None if st.session_state.user.is_superadmin else st.session_state.user.user_id
        )
        
        if recent_candidates:
            recent_data = []
            for candidate in recent_candidates:
                recent_data.append({
                    'Name': candidate.candidate_name,
                    'Resume Code': candidate.candidate_name.replace('Candidate-', ''),
                    'Job': candidate.job.job_title,
                    'Score': f"{candidate.compatibility_score:.1f}%",
                    'Date': candidate.created_at.strftime('%Y-%m-%d'),
                    'Added By': candidate.job.user.username
                })
            st.dataframe(pd.DataFrame(recent_data))
        else:
            st.info("No candidates analyzed yet. Start by creating a job and analyzing resumes.")
    
    def user_management_page(self):
        """User management page (Super Admin only)"""
        st.title("👥 User Management")
        
        if not st.session_state.user.is_superadmin:
            st.error("⛔ Access Denied: Only Super Administrators can access this page.")
            return
        
        tab1, tab2, tab3 = st.tabs(["Add New User", "Manage Existing Users", "System Overview"])
        
        with tab1:
            st.subheader("Add New HR User")
            
            with st.form("add_user_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    new_username = st.text_input("Username*", placeholder="Enter unique username")
                    new_email = st.text_input("Email Address*", placeholder="user@company.com")
                
                with col2:
                    new_password = st.text_input("Password*", type="password", placeholder="Enter secure password")
                    confirm_password = st.text_input("Confirm Password*", type="password", placeholder="Confirm password")
                
                submitted = st.form_submit_button("Create User", type="primary")
                
                if submitted:
                    if not all([new_username, new_email, new_password, confirm_password]):
                        st.error("Please fill in all required fields (*)")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match!")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters long")
                    else:
                        success, message = create_user(
                            username=new_username,
                            password=new_password,
                            email=new_email,
                            is_superadmin=False,
                            created_by=st.session_state.user.user_id
                        )
                        if success:
                            st.success(f"✅ User '{new_username}' created successfully!")
                        else:
                            st.error(f"❌ {message}")
        
        with tab2:
            st.subheader("Manage Existing Users")
            
            users = get_all_users(st.session_state.user.user_id)
            
            if users:
                # Prepare user data for display
                user_data = []
                for user in users:
                    user_data.append({
                        'ID': user.user_id,
                        'Username': user.username,
                        'Email': user.email,
                        'Role': 'Super Admin' if user.is_superadmin else 'HR User',
                        'Status': 'Active' if user.is_active else 'Inactive',
                        'Created By': 'System' if user.created_by is None else 'Another Admin',
                        'Last Login': user.last_login.strftime('%Y-%m-%d %H:%M') if user.last_login else 'Never',
                        'Created Date': user.created_at.strftime('%Y-%m-%d')
                    })
                
                df = pd.DataFrame(user_data)
                st.dataframe(df, use_container_width=True)
                
                # User actions
                st.subheader("User Actions")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Reset Password**")
                    reset_user = st.selectbox("Select User", 
                                            [f"{u['Username']} (ID: {u['ID']})" for u in user_data if u['ID'] != st.session_state.user.user_id],
                                            key="reset_select")
                    new_pass = st.text_input("New Password", type="password", key="new_pass")
                    if st.button("Reset Password", key="reset_btn"):
                        if reset_user and new_pass:
                            user_id = int(reset_user.split("ID: ")[1].split(")")[0])
                            success, message = reset_user_password(user_id, new_pass, True)
                            if success:
                                st.success(f"✅ Password reset for user ID {user_id}")
                            else:
                                st.error(f"❌ {message}")
                        else:
                            st.warning("Please select a user and enter new password")
                
                with col2:
                    st.write("**Deactivate User**")
                    deactivate_user_select = st.selectbox("Select User", 
                                                         [f"{u['Username']} (ID: {u['ID']})" for u in user_data 
                                                          if u['ID'] != st.session_state.user.user_id and u['Status'] == 'Active'],
                                                         key="deactivate_select")
                    if st.button("Deactivate User", type="secondary", key="deactivate_btn"):
                        if deactivate_user_select:
                            user_id = int(deactivate_user_select.split("ID: ")[1].split(")")[0])
                            success, message = deactivate_user(user_id, st.session_state.user.user_id)
                            if success:
                                st.success(f"✅ User ID {user_id} deactivated")
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                        else:
                            st.warning("Please select a user to deactivate")
                
                with col3:
                    st.write("**Activate User**")
                    activate_user_select = st.selectbox("Select User", 
                                                       [f"{u['Username']} (ID: {u['ID']})" for u in user_data 
                                                        if u['ID'] != st.session_state.user.user_id and u['Status'] == 'Inactive'],
                                                       key="activate_select")
                    if st.button("Activate User", type="secondary", key="activate_btn"):
                        if activate_user_select:
                            user_id = int(activate_user_select.split("ID: ")[1].split(")")[0])
                            success, message = activate_user(user_id, st.session_state.user.user_id)
                            if success:
                                st.success(f"✅ User ID {user_id} activated")
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                        else:
                            st.warning("Please select a user to activate")
            
            else:
                st.info("No users found in the system.")
        
        with tab3:
            st.subheader("System Overview")
            
            # System statistics
            users = get_all_users(st.session_state.user.user_id)
            total_users = len(users)
            active_users = len([u for u in users if u.is_active])
            superadmins = len([u for u in users if u.is_superadmin])
            total_jobs, total_candidates = get_user_stats()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Users", total_users)
                st.metric("Active Users", active_users)
            with col2:
                st.metric("Super Admins", superadmins)
                st.metric("Total Jobs", total_jobs)
            with col3:
                st.metric("Total Candidates", total_candidates)
                st.metric("System Status", "Online", "Active")
            
            # Recent system activity
            st.subheader("Recent System Activity")
            recent_logins = [u for u in users if u.last_login is not None]
            recent_logins.sort(key=lambda x: x.last_login, reverse=True)
            recent_logins = recent_logins[:10]
            
            if recent_logins:
                login_data = []
                for user in recent_logins:
                    login_data.append({
                        'Username': user.username,
                        'Role': 'Super Admin' if user.is_superadmin else 'HR User',
                        'Last Login': user.last_login.strftime('%Y-%m-%d %H:%M:%S'),
                        'Status': 'Active' if user.is_active else 'Inactive'
                    })
                st.dataframe(pd.DataFrame(login_data))
    
    def manage_jobs_page(self):
        """Job management page"""
        st.title("💼 Manage Jobs")
        
        tab1, tab2 = st.tabs(["Create New Job", "View Existing Jobs"])
        
        with tab1:
            st.subheader("Create New Job Description")
            
            with st.form("job_form"):
                job_title = st.text_input("Job Title*")
                required_skills = st.text_area("Required Skills*", 
                    placeholder="Enter skills separated by commas\ne.g., Python, Machine Learning, SQL, Communication")
                qualifications = st.text_area("Qualifications",
                    placeholder="Required education, certifications, etc.")
                experience_required = st.selectbox("Experience Required", 
                    ["Not specified", "Entry Level", "1-3 years", "3-5 years", "5+ years"])
                
                submitted = st.form_submit_button("Create Job")
                
                if submitted:
                    if job_title and required_skills:
                        session = get_session()
                        try:
                            job = JobDescription(
                                user_id=st.session_state.user.user_id,
                                job_title=job_title,
                                required_skills=required_skills,
                                qualifications=qualifications,
                                experience_required=experience_required
                            )
                            session.add(job)
                            session.commit()
                            st.success(f"Job '{job_title}' created successfully!")
                        except Exception as e:
                            st.error(f"Error creating job: {str(e)}")
                        finally:
                            close_session()
                    else:
                        st.error("Please fill in all required fields (marked with *)")
        
        with tab2:
            st.subheader("Existing Jobs")
            
            session = get_session()
            try:
                if st.session_state.user.is_superadmin:
                    jobs = session.query(JobDescription).all()
                else:
                    jobs = session.query(JobDescription).filter_by(user_id=st.session_state.user.user_id).all()
                
                if jobs:
                    for job in jobs:
                        # Get fresh job data with relationships to avoid detached instance
                        fresh_job = session.query(JobDescription).filter_by(job_id=job.job_id).first()
                        username = fresh_job.user.username if fresh_job and fresh_job.user else "Unknown"
                        
                        with st.expander(f"📋 {fresh_job.job_title} (Created by: {username})"):
                            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                            
                            with col1:
                                st.write("**Required Skills:**")
                                st.write(fresh_job.required_skills)
                                if fresh_job.qualifications:
                                    st.write("**Qualifications:**")
                                    st.write(fresh_job.qualifications)
                            
                            with col2:
                                st.write(f"**Experience:** {fresh_job.experience_required}")
                                candidates_count = len(fresh_job.candidates) if fresh_job.candidates else 0
                                st.write(f"**Candidates:** {candidates_count}")
                                st.write(f"**Created:** {fresh_job.created_at.strftime('%Y-%m-%d')}")
                            
                            with col3:
                                if st.session_state.user.is_superadmin or fresh_job.user_id == st.session_state.user.user_id:
                                    if st.button(f"Delete Job", key=f"delete_{fresh_job.job_id}"):
                                        session.delete(fresh_job)
                                        session.commit()
                                        st.success(f"Job '{fresh_job.job_title}' deleted successfully!")
                                        st.rerun()
                            
                            with col4:
                                if st.session_state.user.is_superadmin or fresh_job.user_id == st.session_state.user.user_id:
                                    if st.button(f"Delete Analysis", key=f"delete_analysis_{fresh_job.job_id}"):
                                        success, message = delete_job_analysis(fresh_job.job_id, st.session_state.user.user_id)
                                        if success:
                                            st.success(message)
                                        else:
                                            st.error(message)
                                        st.rerun()
                else:
                    st.info("No jobs created yet. Create your first job using the form above.")
            finally:
                close_session()
    
    def analyze_resumes_page(self):
        """Resume analysis page"""
        st.title("🔍 Analyze Resumes")
        
        session = get_session()
        try:
            # Get user's jobs (or all jobs for superadmin)
            if st.session_state.user.is_superadmin:
                jobs = session.query(JobDescription).all()
            else:
                jobs = session.query(JobDescription).filter_by(user_id=st.session_state.user.user_id).all()
            
            if not jobs:
                st.warning("Please create a job first before analyzing resumes.")
                return
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Select Job & Upload Resumes")
                selected_job = st.selectbox("Select Job", jobs, format_func=lambda x: f"{x.job_title} (by {x.user.username})")
                
                st.markdown("---")
                st.subheader("Upload Resumes")
                uploaded_files = st.file_uploader(
                    "Choose PDF or DOCX files",
                    type=["pdf", "docx"],
                    accept_multiple_files=True,
                    help="Upload multiple PDF or DOCX resume files"
                )
                
                # Duplicate handling option
                st.markdown("---")
                st.subheader("Duplicate Handling")
                handle_duplicates = st.radio(
                    "If duplicate resumes found:",
                    ["Skip duplicates", "Overwrite existing analysis"],
                    help="Choose how to handle resumes that have already been analyzed for this job"
                )
                
                if uploaded_files and selected_job:
                    st.info(f"📁 {len(uploaded_files)} file(s) selected")
                    if st.button("🚀 Analyze Resumes", type="primary", use_container_width=True):
                        self.process_uploaded_resumes(uploaded_files, selected_job, handle_duplicates)
                else:
                    st.info("👆 Please select files to upload")
            
            with col2:
                if selected_job:
                    st.subheader(f"Job Details: {selected_job.job_title}")
                    
                    st.write("**Required Skills:**")
                    st.write(selected_job.required_skills)
                    
                    if selected_job.qualifications:
                        st.write("**Qualifications:**")
                        st.write(selected_job.qualifications)
                    
                    st.write("**Experience Level:**", selected_job.experience_required)
                    st.write("**Created by:**", selected_job.user.username)
                    
                    candidates_count = len(selected_job.candidates) if selected_job.candidates else 0
                    st.metric("Candidates Analyzed", candidates_count)
        finally:
            close_session()

    def process_uploaded_resumes(self, uploaded_files, job, handle_duplicates):
        """Process uploaded resumes with comprehensive analysis and duplicate handling"""
        if not uploaded_files:
            st.warning("No files uploaded. Please select resume files.")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        successful_processed = 0
        failed_processed = 0
        skipped_duplicates = 0
        all_reports = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"🔍 Analyzing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Parse resume with enhanced parser
                file_type = uploaded_file.name.split('.')[-1].lower()
                parsed_data = self.resume_parser.parse_resume(tmp_path, file_type)
                
                if "error" in parsed_data:
                    st.error(f"Error parsing {uploaded_file.name}: {parsed_data['error']}")
                    failed_processed += 1
                    continue
                
                # Check for duplicate candidate
                resume_code = parsed_data.get('resume_code')
                if resume_code and check_duplicate_candidate(job.job_id, resume_code):
                    if handle_duplicates == "Skip duplicates":
                        st.warning(f"⏭️ Skipped duplicate resume: {uploaded_file.name}")
                        skipped_duplicates += 1
                        continue
                    else:
                        st.info(f"🔄 Overwriting analysis for: {uploaded_file.name}")
                
                # Generate comprehensive analysis report
                comprehensive_report = self.scoring_model.generate_comprehensive_report(
                    job.required_skills, parsed_data
                )
                
                # Store CV file in database (handles duplicates internally)
                if resume_code:
                    success, message = store_cv_file(
                        resume_code=resume_code,
                        filename=uploaded_file.name,
                        file_content=uploaded_file.getvalue(),
                        file_type=file_type,
                        user_id=st.session_state.user.user_id
                    )
                    if not success:
                        st.warning(f"Could not store CV file: {message}")
                
                # Store candidate in database
                session = get_session()
                try:
                    # Check if candidate already exists (for overwrite case)
                    existing_candidate = session.query(Candidate).filter_by(
                        job_id=job.job_id,
                        candidate_name=f"Candidate-{resume_code}"
                    ).first()
                    
                    if existing_candidate:
                        # Update existing candidate
                        existing_candidate.compatibility_score = comprehensive_report["job_fit_analysis"]["compatibility_score"]
                        existing_candidate.processed_text = parsed_data.get("processed_text", "")
                        
                        # Delete existing skills and add new ones
                        session.query(Skill).filter_by(candidate_id=existing_candidate.candidate_id).delete()
                        
                        candidate_id = existing_candidate.candidate_id
                        candidate_name = existing_candidate.candidate_name
                    else:
                        # Create new candidate
                        candidate = Candidate(
                            job_id=job.job_id,
                            candidate_name=f"Candidate-{resume_code}",
                            email="",  # No email for confidentiality
                            phone=parsed_data.get("phone_number", ""),
                            compatibility_score=comprehensive_report["job_fit_analysis"]["compatibility_score"],
                            resume_file_path=uploaded_file.name,
                            processed_text=parsed_data.get("processed_text", "")
                        )
                        session.add(candidate)
                        session.flush()
                        
                        candidate_id = candidate.candidate_id
                        candidate_name = candidate.candidate_name
                    
                    # Store skills
                    skills_list = parsed_data.get("skills_list", [])
                    for skill_name, category, confidence in skills_list:
                        skill = Skill(
                            candidate_id=candidate_id,
                            skill_name=skill_name,
                            confidence_score=confidence
                        )
                        session.add(skill)
                    
                    session.commit()
                    successful_processed += 1
                    
                    # Store report for display
                    all_reports.append({
                        'candidate_name': candidate_name,
                        'file_name': uploaded_file.name,
                        'report': comprehensive_report,
                        'parsed_data': parsed_data,
                        'resume_code': resume_code
                    })
                    
                except Exception as e:
                    session.rollback()
                    st.error(f"Database error for {uploaded_file.name}: {str(e)}")
                    failed_processed += 1
                finally:
                    close_session()
                
            except Exception as e:
                failed_processed += 1
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("✅ Analysis complete!")
        
        # Display comprehensive reports
        with results_container:
            if successful_processed > 0:
                st.success(f"✅ Successfully analyzed {successful_processed} resumes!")
            
            if skipped_duplicates > 0:
                st.info(f"⏭️ Skipped {skipped_duplicates} duplicate resumes")
            
            if failed_processed > 0:
                st.error(f"❌ Failed to process {failed_processed} resumes")
            
            # Show detailed reports for each candidate
            for report_data in all_reports:
                self.display_detailed_report(report_data, job)

    def display_detailed_report(self, report_data, job):
        """Display comprehensive analysis report for a candidate"""
        candidate_name = report_data['candidate_name']
        report = report_data['report']
        parsed_data = report_data['parsed_data']
        resume_code = report_data['resume_code']
        
        with st.expander(f"📊 Detailed Analysis: {candidate_name}", expanded=True):
            # Confidential Code Display
            st.markdown(f"<div class='confidential-code'>Confidential Resume Code: {resume_code}</div>", unsafe_allow_html=True)
            st.caption("🔒 Use this code with phone number for confidential candidate lookup")
            
            # Overall Score Card
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                score = report['job_fit_analysis']['compatibility_score']
                color = "green" if score >= 70 else "orange" if score >= 50 else "red"
                st.metric("Job Fit Score", f"{score}%", delta=report['quality_analysis']['overall_score'], 
                         delta_color="normal")
            
            with col2:
                st.metric("Resume Quality", f"{report['quality_analysis']['overall_score']}%")
            
            with col3:
                priority = report['interview_priority']
                st.metric("Interview Priority", priority)
            
            with col4:
                st.metric("ATS Compatibility", f"{report['quality_analysis']['ats_compatibility']}%")
            
            # Tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "🎯 Job Fit", "📝 Resume Quality", "👤 Candidate Details"])
            
            with tab1:
                self.display_summary_tab(report, parsed_data, job)
            
            with tab2:
                self.display_job_fit_tab(report)
            
            with tab3:
                self.display_quality_tab(report)
            
            with tab4:
                self.display_candidate_details_tab(parsed_data)

    def display_summary_tab(self, report, parsed_data, job):
        """Display summary tab content"""
        st.subheader("Executive Summary")
        
        # Recommendation
        reco = report['recommendation']
        if "STRONG RECOMMEND" in reco or "EXCEPTIONAL" in reco or "TOP CANDIDATE" in reco:
            st.success(f"🎯 {reco}")
        elif "RECOMMEND" in reco:
            st.info(f"💡 {reco}")
        elif "CONSIDER" in reco or "PROMISING" in reco:
            st.warning(f"⚠️ {reco}")
        else:
            st.error(f"❌ {reco}")
        
        # Key metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Matching Skills:**")
            matching_skills = report['job_fit_analysis']['matching_skills'][:5]
            for skill in matching_skills:
                st.write(f"✅ {skill}")
            
            if report['job_fit_analysis']['missing_skills']:
                st.write("**Skills to Develop:**")
                for skill in report['job_fit_analysis']['missing_skills'][:3]:
                    st.write(f"📚 {skill}")
        
        with col2:
            st.write("**Resume Strengths:**")
            for strength in report['quality_analysis']['strengths'][:3]:
                st.write(f"🌟 {strength}")
            
            st.write("**Areas for Improvement:**")
            for improvement in report['quality_analysis']['improvements'][:2]:
                st.write(f"🔧 {improvement}")
        
        # Interview Questions
        if report.get('interview_questions'):
            st.subheader("Suggested Interview Questions")
            for i, question in enumerate(report['interview_questions'][:3], 1):
                st.write(f"{i}. {question}")

    def display_job_fit_tab(self, report):
        """Display job fit analysis"""
        fit_analysis = report['job_fit_analysis']
        
        # Score breakdown
        st.subheader("Job Fit Breakdown")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Skill Match", f"{fit_analysis['skill_match']}%")
        with col2:
            st.metric("Experience Match", f"{fit_analysis['experience_match']}%")
        with col3:
            st.metric("Education Match", f"{fit_analysis['education_match']}%")
        
        # Fit analysis text
        st.write("**Analysis:**")
        st.info(fit_analysis['fit_analysis'])
        
        # Skills visualization
        if fit_analysis['matching_skills'] or fit_analysis['missing_skills']:
            st.write("**Skill Match Details:**")
            skills_df = pd.DataFrame({
                'Category': ['Matching Skills', 'Missing Skills'],
                'Count': [len(fit_analysis['matching_skills']), len(fit_analysis['missing_skills'])]
            })
            
            fig = px.pie(skills_df, values='Count', names='Category', 
                         title="Skill Match Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Skill category breakdown
        if 'skill_match_breakdown' in fit_analysis:
            st.subheader("Skill Category Analysis")
            category_data = []
            for category, data in fit_analysis['skill_match_breakdown'].items():
                category_data.append({
                    'Category': category.replace('_', ' ').title(),
                    'Match Percentage': data.get('match_percentage', 0),
                    'Matched Skills': len(data.get('matched_skills', []))
                })
            
            if category_data:
                category_df = pd.DataFrame(category_data)
                fig = px.bar(category_df, x='Category', y='Match Percentage',
                           title="Skill Match by Category", color='Match Percentage')
                st.plotly_chart(fig, use_container_width=True)

    def display_quality_tab(self, report):
        """Display resume quality analysis"""
        quality = report['quality_analysis']
        
        st.subheader("Resume Quality Assessment")
        
        # Section scores
        sections = list(quality['section_scores'].keys())
        scores = list(quality['section_scores'].values())
        
        fig = px.bar(x=sections, y=scores, title="Section Quality Scores",
                     labels={'x': 'Section', 'y': 'Score'},
                     color=scores, color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed feedback
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Strengths:**")
            for strength in quality['strengths']:
                st.success(f"✅ {strength}")
        
        with col2:
            st.write("**Improvements:**")
            for improvement in quality['improvements']:
                st.warning(f"📝 {improvement}")

    def display_candidate_details_tab(self, parsed_data):
        """Display detailed candidate information (confidential)"""
        st.subheader("Extracted Information")
        
        # Confidential notice
        st.info("🔒 **Confidential Analysis**: Personal identification details are protected. Use resume code for candidate lookup.")
        
        # Contact Information (limited for confidentiality)
        st.write("**Contact Reference:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Phone Reference:** {parsed_data.get('phone_number', 'Not available')}")
        
        with col2:
            st.write(f"**Resume Code:** {parsed_data.get('resume_code', 'N/A')}")
        
        # Qualifications
        qualifications = parsed_data.get('qualifications', {})
        if qualifications.get('all_qualifications'):
            st.write("**Education:**")
            for i, qual in enumerate(qualifications['all_qualifications'][:3]):
                st.write(f"{i+1}. **{qual.get('level', 'Qualification').title()}**")
                if qual.get('institution'):
                    st.write(f"   Institution: {qual['institution']}")
                if qual.get('field'):
                    st.write(f"   Field: {qual['field']}")
                st.write("")
        
        # Experience
        experience = parsed_data.get('experience', {})
        if experience.get('career_progression'):
            st.write("**Experience Summary:**")
            st.write(f"Total Experience: {experience.get('total_experience_years', 0)} years")
            st.write(f"Experience Level: {experience.get('experience_level', 'Not specified')}")
            
            if experience.get('industry_sectors'):
                st.write(f"Industry Sectors: {', '.join(experience['industry_sectors'])}")
        
        # Skills
        skills = parsed_data.get('skills', {})
        if skills.get('top_skills'):
            st.write("**Top Skills:**")
            top_skills_display = [skill['skill'] for skill in skills['top_skills'][:10]]
            st.write(", ".join(top_skills_display))
        
        # Achievements
        awards = parsed_data.get('awards_certifications', {})
        total_achievements = sum(len(awards.get(category, [])) for category in ['awards', 'certifications', 'honors'])
        if total_achievements > 0:
            st.write(f"**Achievements & Certifications:** {total_achievements} found")

    def cv_lookup_page(self):
        """Page to lookup and view CVs by resume code"""
        st.title("📁 CV Lookup System")
        
        st.info("🔒 **Confidential System**: Look up CVs using resume codes and phone numbers")
        
        tab1, tab2 = st.tabs(["Lookup by Resume Code", "Lookup by Phone Number"])
        
        with tab1:
            st.subheader("Lookup by Resume Code")
            resume_code = st.text_input("Enter Resume Code", placeholder="e.g., CV-6BDEC089-1017", key="resume_code_lookup")
            
            if st.button("🔍 Lookup CV by Code", type="primary", key="lookup_code_btn"):
                if resume_code:
                    self.lookup_and_display_cv_by_code(resume_code)
                else:
                    st.warning("Please enter a resume code")
        
        with tab2:
            st.subheader("Lookup by Phone Number")
            phone_number = st.text_input("Enter Phone Number", placeholder="e.g., 0771234567", key="phone_lookup")
            
            if st.button("🔍 Lookup CV by Phone", type="primary", key="lookup_phone_btn"):
                if phone_number:
                    self.lookup_and_display_cv_by_phone(phone_number)
                else:
                    st.warning("Please enter a phone number")
        
        # Show recent CVs for quick access
        st.subheader("Recent CVs")
        session = get_session()
        try:
            if st.session_state.user.is_superadmin:
                candidates = session.query(Candidate).order_by(Candidate.created_at.desc()).limit(10).all()
            else:
                candidates = session.query(Candidate).join(JobDescription).filter(
                    JobDescription.user_id == st.session_state.user.user_id
                ).order_by(Candidate.created_at.desc()).limit(10).all()
            
            if candidates:
                for candidate in candidates:
                    resume_code = candidate.candidate_name.replace('Candidate-', '')
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{candidate.candidate_name}**")
                        st.write(f"Job: {candidate.job.job_title}")
                    with col2:
                        st.write(f"Score: {candidate.compatibility_score}%")
                    with col3:
                        if st.button(f"View", key=f"view_{resume_code}"):
                            self.lookup_and_display_cv_by_code(resume_code)
                    st.write("---")
            else:
                st.info("No candidates analyzed yet")
        finally:
            close_session()

    def lookup_and_display_cv_by_code(self, resume_code):
        """Lookup and display CV by resume code"""
        cv_record = get_cv_by_resume_code(resume_code, st.session_state.user.user_id)
        
        if cv_record:
            self.display_cv_file(cv_record, resume_code)
        else:
            st.error("❌ CV not found or you don't have permission to access this CV")
    
    def lookup_and_display_cv_by_phone(self, phone_number):
        """Lookup and display CV by phone number"""
        cv_record, candidate = get_cv_by_phone_number(phone_number, st.session_state.user.user_id)
        
        if cv_record and candidate:
            st.success(f"✅ CV found for phone number: {phone_number}")
            self.display_cv_file(cv_record, candidate.candidate_name.replace('Candidate-', ''))
        else:
            st.error("❌ No CV found for this phone number or you don't have permission to access it")

    def display_cv_file(self, cv_record, resume_code):
        """Display the actual CV file"""
        st.markdown(f"<div class='cv-viewer'>", unsafe_allow_html=True)
        st.subheader(f"📄 Original CV: {cv_record.original_filename}")
        st.markdown(f"**Resume Code:** `{resume_code}`")
        st.markdown(f"**Uploaded:** {cv_record.uploaded_at.strftime('%Y-%m-%d %H:%M')}")
        
        # Display based on file type
        if cv_record.file_type == 'pdf':
            # For PDF, provide download and show preview
            b64_pdf = base64.b64encode(cv_record.file_content).decode()
            pdf_display = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
            
            # Download button
            st.download_button(
                label="📥 Download PDF",
                data=cv_record.file_content,
                file_name=cv_record.original_filename,
                mime="application/pdf"
            )
        
        elif cv_record.file_type == 'docx':
            # For DOCX, provide download and show message
            st.info("📝 DOCX file - Download to view original document")
            
            # Download button
            st.download_button(
                label="📥 Download DOCX",
                data=cv_record.file_content,
                file_name=cv_record.original_filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        
        else:
            st.warning(f"Unsupported file type: {cv_record.file_type}")
        
        st.markdown("</div>", unsafe_allow_html=True)

    def view_reports_page(self):
        """Enhanced Reports and analytics page with comprehensive insights"""
        st.title("📈 Advanced Candidate Analytics")
        
        session = get_session()
        try:
            # Get jobs based on user role
            if st.session_state.user.is_superadmin:
                jobs = session.query(JobDescription).all()
            else:
                jobs = session.query(JobDescription).filter_by(user_id=st.session_state.user.user_id).all()
            
            if not jobs:
                st.info("No jobs available. Create a job first.")
                return
            
            selected_job = st.selectbox("Select Job for Analysis", jobs, 
                                       format_func=lambda x: f"{x.job_title} (by {x.user.username})")
            
            if selected_job:
                candidates = session.query(Candidate).filter_by(job_id=selected_job.job_id).order_by(
                    Candidate.compatibility_score.desc()
                ).all()
                
                if candidates:
                    # Enhanced candidate data preparation
                    candidate_data = []
                    skill_frequency = {}
                    experience_levels = {}
                    score_distribution = {'Excellent (80-100)': 0, 'Good (60-79)': 0, 'Average (40-59)': 0, 'Poor (0-39)': 0}
                    
                    for candidate in candidates:
                        # Collect skills for frequency analysis
                        skills = [skill.skill_name for skill in candidate.skills]
                        for skill in skills:
                            skill_frequency[skill] = skill_frequency.get(skill, 0) + 1
                        
                        # Determine experience level (simplified)
                        exp_level = "Not Specified"
                        if candidate.processed_text:
                            if "years" in candidate.processed_text.lower():
                                exp_level = "Experienced"
                            elif "entry" in candidate.processed_text.lower() or "junior" in candidate.processed_text.lower():
                                exp_level = "Entry Level"
                        experience_levels[exp_level] = experience_levels.get(exp_level, 0) + 1
                        
                        # Score distribution
                        score = candidate.compatibility_score
                        if score >= 80:
                            score_distribution['Excellent (80-100)'] += 1
                        elif score >= 60:
                            score_distribution['Good (60-79)'] += 1
                        elif score >= 40:
                            score_distribution['Average (40-59)'] += 1
                        else:
                            score_distribution['Poor (0-39)'] += 1
                        
                        candidate_data.append({
                            'Rank': len(candidate_data) + 1,
                            'Name': candidate.candidate_name,
                            'Resume Code': candidate.candidate_name.replace('Candidate-', ''),
                            'Score': f"{candidate.compatibility_score:.1f}%",
                            'Top Skills': ', '.join(skills[:3]) if skills else 'Not specified',
                            'Phone Reference': candidate.phone,
                            'Status': 'Top Candidate' if candidate.compatibility_score >= 80 else 
                                     'Good Match' if candidate.compatibility_score >= 60 else 
                                     'Average' if candidate.compatibility_score >= 40 else 'Low Match',
                            'Experience Level': exp_level
                        })
                    
                    df = pd.DataFrame(candidate_data)
                    
                    # Enhanced layout with multiple tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "👥 Candidate Ranking", "🔍 Deep Analysis", "📋 Export Data"])
                    
                    with tab1:
                        st.subheader("📊 Recruitment Dashboard")
                        
                        # Key Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Candidates", len(candidates))
                        with col2:
                            avg_score = sum(c.compatibility_score for c in candidates) / len(candidates)
                            st.metric("Average Score", f"{avg_score:.1f}%")
                        with col3:
                            top_candidates = len([c for c in candidates if c.compatibility_score >= 80])
                            st.metric("Top Candidates", top_candidates)
                        with col4:
                            response_rate = f"{(len(candidates) / max(len(candidates), 1)) * 100:.1f}%"
                            st.metric("Analysis Completion", response_rate)
                        
                        # Visualizations in columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Score distribution
                            st.subheader("Score Distribution")
                            fig_score = px.pie(
                                values=list(score_distribution.values()),
                                names=list(score_distribution.keys()),
                                title="Candidate Score Distribution",
                                color=list(score_distribution.keys()),
                                color_discrete_map={
                                    'Excellent (80-100)': '#00CC96',
                                    'Good (60-79)': '#FFA15C', 
                                    'Average (40-59)': '#FFD700',
                                    'Poor (0-39)': '#EF553B'
                                }
                            )
                            st.plotly_chart(fig_score, use_container_width=True)
                            
                            # Experience level distribution
                            if experience_levels:
                                st.subheader("Experience Level Distribution")
                                fig_exp = px.bar(
                                    x=list(experience_levels.keys()),
                                    y=list(experience_levels.values()),
                                    title="Candidate Experience Levels",
                                    labels={'x': 'Experience Level', 'y': 'Count'},
                                    color=list(experience_levels.values()),
                                    color_continuous_scale='Viridis'
                                )
                                st.plotly_chart(fig_exp, use_container_width=True)
                        
                        with col2:
                            # Top skills analysis
                            if skill_frequency:
                                st.subheader("Top 10 Skills in Candidate Pool")
                                top_skills = sorted(skill_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
                                skills_df = pd.DataFrame(top_skills, columns=['Skill', 'Frequency'])
                                fig_skills = px.bar(
                                    skills_df,
                                    x='Frequency',
                                    y='Skill',
                                    orientation='h',
                                    title="Most Common Skills",
                                    color='Frequency',
                                    color_continuous_scale='Blues'
                                )
                                st.plotly_chart(fig_skills, use_container_width=True)
                            
                            # Score trend by candidate rank
                            st.subheader("Score Distribution by Rank")
                            fig_trend = px.scatter(
                                df,
                                x='Rank',
                                y=df['Score'].str.replace('%', '').astype(float),
                                color=df['Score'].str.replace('%', '').astype(float),
                                size=[10] * len(df),
                                title="Candidate Scores by Ranking",
                                labels={'y': 'Score (%)', 'color': 'Score'},
                                color_continuous_scale='RdYlGn'
                            )
                            fig_trend.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
                            st.plotly_chart(fig_trend, use_container_width=True)
                    
                    with tab2:
                        st.subheader("👥 Detailed Candidate Ranking")
                        
                        # Enhanced filtering options
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            score_filter = st.selectbox("Filter by Score Range", 
                                                      ["All", "Top (80-100%)", "Good (60-79%)", "Average (40-59%)", "Low (0-39%)"])
                        with col2:
                            status_filter = st.selectbox("Filter by Status", 
                                                        ["All", "Top Candidate", "Good Match", "Average", "Low Match"])
                        with col3:
                            experience_filter = st.selectbox("Filter by Experience", 
                                                           ["All", "Experienced", "Entry Level", "Not Specified"])
                        
                        # Apply filters
                        filtered_df = df.copy()
                        if score_filter != "All":
                            if score_filter == "Top (80-100%)":
                                filtered_df = filtered_df[filtered_df['Score'].str.replace('%', '').astype(float) >= 80]
                            elif score_filter == "Good (60-79%)":
                                filtered_df = filtered_df[(filtered_df['Score'].str.replace('%', '').astype(float) >= 60) & 
                                                         (filtered_df['Score'].str.replace('%', '').astype(float) < 80)]
                            elif score_filter == "Average (40-59%)":
                                filtered_df = filtered_df[(filtered_df['Score'].str.replace('%', '').astype(float) >= 40) & 
                                                         (filtered_df['Score'].str.replace('%', '').astype(float) < 60)]
                            elif score_filter == "Low (0-39%)":
                                filtered_df = filtered_df[filtered_df['Score'].str.replace('%', '').astype(float) < 40]
                        
                        if status_filter != "All":
                            filtered_df = filtered_df[filtered_df['Status'] == status_filter]
                        
                        if experience_filter != "All":
                            filtered_df = filtered_df[filtered_df['Experience Level'] == experience_filter]
                        
                        # Display filtered results
                        st.dataframe(filtered_df, use_container_width=True)
                        
                        # Summary of filtered results
                        st.info(f"📋 Showing {len(filtered_df)} of {len(df)} candidates")
                    
                    with tab3:
                        st.subheader("🔍 Deep Analysis & Insights")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Skill gap analysis
                            st.subheader("🔄 Skill Gap Analysis")
                            job_skills = [skill.strip().lower() for skill in selected_job.required_skills.split(',')]
                            candidate_skills_flat = [skill.lower() for skills_list in skill_frequency.keys() for skill in skills_list.split(', ')]
                            
                            missing_skills = set(job_skills) - set(candidate_skills_flat)
                            strong_skills = set(candidate_skills_flat) & set(job_skills)
                            
                            st.write("**Strongly Covered Skills:**")
                            for skill in list(strong_skills)[:5]:
                                st.success(f"✅ {skill.title()}")
                            
                            if missing_skills:
                                st.write("**Skills Needing Attention:**")
                                for skill in list(missing_skills)[:5]:
                                    st.error(f"❌ {skill.title()}")
                            else:
                                st.success("🎉 All required skills are covered in the candidate pool!")
                        
                        with col2:
                            # Recruitment recommendations
                            st.subheader("💡 Recruitment Recommendations")
                            
                            if avg_score >= 70:
                                st.success("**Strong Candidate Pool** - Proceed with interviews for top candidates")
                            elif avg_score >= 50:
                                st.warning("**Moderate Candidate Pool** - Consider additional sourcing or review screening criteria")
                            else:
                                st.error("**Weak Candidate Pool** - Revise job requirements or expand sourcing channels")
                            
                            if top_candidates >= 3:
                                st.success(f"**{top_candidates} Top Candidates** - Sufficient high-quality candidates for selection")
                            else:
                                st.warning(f"**Only {top_candidates} Top Candidates** - Consider additional sourcing")
                            
                            # Experience insights
                            if experience_levels.get("Experienced", 0) >= len(candidates) * 0.5:
                                st.success("**Experience-Rich Pool** - Strong industry experience available")
                            else:
                                st.info("**Mixed Experience Levels** - Consider both experienced and potential candidates")
                        
                        # Candidate progression timeline
                        st.subheader("📈 Candidate Analysis Timeline")
                        if candidates:
                            timeline_data = []
                            for candidate in candidates:
                                timeline_data.append({
                                    'Candidate': candidate.candidate_name,
                                    'Date': candidate.created_at,
                                    'Score': candidate.compatibility_score,
                                    'Status': 'Top' if candidate.compatibility_score >= 80 else 'Good' if candidate.compatibility_score >= 60 else 'Average'
                                })
                            
                            timeline_df = pd.DataFrame(timeline_data)
                            timeline_df['Date'] = pd.to_datetime(timeline_df['Date'])
                            timeline_df = timeline_df.sort_values('Date')
                            
                            fig_timeline = px.scatter(
                                timeline_df,
                                x='Date',
                                y='Score',
                                color='Status',
                                size=[15] * len(timeline_df),
                                title="Candidate Analysis Over Time",
                                color_discrete_map={
                                    'Top': '#00CC96',
                                    'Good': '#FFA15C',
                                    'Average': '#EF553B'
                                }
                            )
                            st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    with tab4:
                        st.subheader("📋 Export & Share Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Export candidate data
                            st.write("**Export Candidate Data**")
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Full CSV Report",
                                data=csv,
                                file_name=f"candidates_{selected_job.job_title}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv"
                            )
                            
                            # Export filtered data
                            if len(filtered_df) < len(df):
                                filtered_csv = filtered_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Filtered CSV",
                                    data=filtered_csv,
                                    file_name=f"filtered_candidates_{selected_job.job_title}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                    mime="text/csv"
                                )
                        
                        with col2:
                            # Summary report
                            st.write("**Generate Summary Report**")
                            if st.button("📄 Generate Executive Summary"):
                                summary_report = self._generate_executive_summary(
                                    selected_job, candidates, avg_score, top_candidates, skill_frequency, experience_levels
                                )
                                st.text_area("Executive Summary", summary_report, height=200)
                        
                        # Quick stats for sharing
                        st.subheader("📊 Quick Statistics for Sharing")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Analyzed", len(candidates))
                            st.metric("Top Candidates", top_candidates)
                        with col2:
                            st.metric("Average Score", f"{avg_score:.1f}%")
                            st.metric("Success Rate", f"{(top_candidates/len(candidates))*100:.1f}%" if candidates else "0%")
                        with col3:
                            st.metric("Skill Coverage", f"{(len(strong_skills)/len(job_skills))*100:.1f}%" if job_skills else "100%")
                            st.metric("Experience Diversity", f"{len(experience_levels)} levels")
                
                else:
                    st.info("No candidates analyzed for this job yet. Start by analyzing resumes in the 'Analyze Resumes' section.")
        finally:
            close_session()

    def _generate_executive_summary(self, job, candidates, avg_score, top_candidates, skill_frequency, experience_levels):
        """Generate an executive summary report"""
        summary = f"""
EXECUTIVE SUMMARY - {job.job_title}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

OVERVIEW:
• Total Candidates Analyzed: {len(candidates)}
• Average Compatibility Score: {avg_score:.1f}%
• Top Candidates (80%+): {top_candidates}
• Recruitment Success Rate: {(top_candidates/len(candidates))*100:.1f}%

SKILLS ANALYSIS:
• Total Unique Skills Identified: {len(skill_frequency)}
• Top 3 Most Common Skills: {', '.join([skill for skill, _ in sorted(skill_frequency.items(), key=lambda x: x[1], reverse=True)[:3]])}

EXPERIENCE DISTRIBUTION:
{chr(10).join([f"• {level}: {count} candidates" for level, count in experience_levels.items()])}

RECOMMENDATIONS:
{"• STRONG POOL: Proceed with interviewing top candidates immediately" if avg_score >= 70 else 
"• MODERATE POOL: Consider additional screening or revised criteria" if avg_score >= 50 else 
"• WEAK POOL: Expand sourcing channels or revise job requirements"}

NEXT STEPS:
1. Review top {min(5, top_candidates)} candidates in detail
2. Schedule interviews for candidates scoring 80% and above
3. Consider skill development for internal candidates if gaps exist

This analysis provides a comprehensive view of the candidate pool suitability for the {job.job_title} position.
"""
        return summary

    def user_profile_page(self):
        """User profile and settings page"""
        st.title("👤 User Profile")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Profile Information")
            st.write(f"**Username:** {st.session_state.user.username}")
            st.write(f"**Email:** {st.session_state.user.email}")
            st.write(f"**Role:** {'Super Administrator' if st.session_state.user.is_superadmin else 'HR User'}")
            st.write(f"**Account Created:** {st.session_state.user.created_at.strftime('%Y-%m-%d')}")
            st.write(f"**Last Login:** {st.session_state.user.last_login.strftime('%Y-%m-%d %H:%M') if st.session_state.user.last_login else 'Never'}")
            st.write(f"**Status:** {'Active' if st.session_state.user.is_active else 'Inactive'}")
        
        with col2:
            st.subheader("Change Password")
            
            with st.form("change_password_form"):
                current_password = st.text_input("Current Password", type="password")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm New Password", type="password")
                
                submitted = st.form_submit_button("Update Password")
                
                if submitted:
                    if not all([current_password, new_password, confirm_password]):
                        st.error("Please fill in all fields")
                    elif new_password != confirm_password:
                        st.error("New passwords do not match")
                    elif len(new_password) < 6:
                        st.error("New password must be at least 6 characters long")
                    else:
                        # Verify current password
                        session = get_session()
                        try:
                            user = session.query(User).filter_by(user_id=st.session_state.user.user_id).first()
                            if user and bcrypt.checkpw(current_password.encode('utf-8'), user.hashed_password.encode('utf-8')):
                                success, message = reset_user_password(st.session_state.user.user_id, new_password)
                                if success:
                                    st.success("✅ Password updated successfully!")
                                else:
                                    st.error(f"❌ {message}")
                            else:
                                st.error("Current password is incorrect")
                        finally:
                            close_session()
    
    def run(self):
        """Main application runner"""
        if self.login_page():
            # Sidebar navigation
            st.sidebar.title(f"Welcome, {st.session_state.user.username}")
            
            if st.session_state.user.is_superadmin:
                st.sidebar.markdown('<span class="superadmin-badge">SUPER ADMIN</span>', unsafe_allow_html=True)
            
            # Navigation options based on user role
            if st.session_state.user.is_superadmin:
                menu_options = {
                    "📊 Dashboard": self.dashboard_page,
                    "👥 User Management": self.user_management_page,
                    "💼 Manage Jobs": self.manage_jobs_page,
                    "🔍 Analyze Resumes": self.analyze_resumes_page,
                    "📁 CV Lookup": self.cv_lookup_page,
                    "📈 View Reports": self.view_reports_page,
                    "👤 Profile": self.user_profile_page
                }
            else:
                menu_options = {
                    "📊 Dashboard": self.dashboard_page,
                    "💼 Manage Jobs": self.manage_jobs_page,
                    "🔍 Analyze Resumes": self.analyze_resumes_page,
                    "📁 CV Lookup": self.cv_lookup_page,
                    "📈 View Reports": self.view_reports_page,
                    "👤 Profile": self.user_profile_page
                }
            
            selected_menu = st.sidebar.radio("Navigation", list(menu_options.keys()))
            
            # Logout button
            st.sidebar.markdown("---")
            if st.sidebar.button("🚪 Logout", use_container_width=True):
                st.session_state.user = None
                st.rerun()
            
            # Execute selected page
            menu_options[selected_menu]()

# Run the application
if __name__ == "__main__":
    app = ResumeAnalyzerApp()
    app.run()
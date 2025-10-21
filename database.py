from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey, Boolean, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import bcrypt
import threading

Base = declarative_base()

# Thread-local session
thread_local = threading.local()

class User(Base):
    __tablename__ = 'users'
    
    user_id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    email = Column(String(100))
    is_superadmin = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_by = Column(Integer, ForeignKey('users.user_id'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    job_descriptions = relationship("JobDescription", back_populates="user")
    created_users = relationship("User", remote_side=[user_id])
    cv_storage = relationship("CVStorage", back_populates="user")

class JobDescription(Base):
    __tablename__ = 'job_descriptions'
    
    job_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    job_title = Column(String(200), nullable=False)
    required_skills = Column(Text)
    qualifications = Column(Text)
    experience_required = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="job_descriptions")
    candidates = relationship("Candidate", back_populates="job", cascade="all, delete-orphan")

class Candidate(Base):
    __tablename__ = 'candidates'
    
    candidate_id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey('job_descriptions.job_id'))
    candidate_name = Column(String(200))
    email = Column(String(200))
    phone = Column(String(50))
    compatibility_score = Column(Float)
    resume_file_path = Column(String(500))
    processed_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    job = relationship("JobDescription", back_populates="candidates")
    skills = relationship("Skill", back_populates="candidate", cascade="all, delete-orphan")
    educations = relationship("Education", back_populates="candidate", cascade="all, delete-orphan")
    experiences = relationship("WorkExperience", back_populates="candidate", cascade="all, delete-orphan")

class Skill(Base):
    __tablename__ = 'skills'
    
    skill_id = Column(Integer, primary_key=True)
    candidate_id = Column(Integer, ForeignKey('candidates.candidate_id'))
    skill_name = Column(String(100))
    confidence_score = Column(Float)
    
    candidate = relationship("Candidate", back_populates="skills")

class Education(Base):
    __tablename__ = 'education'
    
    education_id = Column(Integer, primary_key=True)
    candidate_id = Column(Integer, ForeignKey('candidates.candidate_id'))
    institution = Column(String(200))
    degree = Column(String(200))
    field_of_study = Column(String(200))
    start_date = Column(String(50))
    end_date = Column(String(50))
    
    candidate = relationship("Candidate", back_populates="educations")

class WorkExperience(Base):
    __tablename__ = 'work_experience'
    
    experience_id = Column(Integer, primary_key=True)
    candidate_id = Column(Integer, ForeignKey('candidates.candidate_id'))
    company = Column(String(200))
    position = Column(String(200))
    start_date = Column(String(50))
    end_date = Column(String(50))
    description = Column(Text)
    
    candidate = relationship("Candidate", back_populates="experiences")

class CVStorage(Base):
    __tablename__ = 'cv_storage'
    
    id = Column(Integer, primary_key=True)
    resume_code = Column(String(50), unique=True, nullable=False)
    original_filename = Column(String(500), nullable=False)
    file_content = Column(LargeBinary, nullable=False)
    file_type = Column(String(10), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    
    user = relationship("User", back_populates="cv_storage")

# Database setup
engine = create_engine('sqlite:///resume_analyzer.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def get_session():
    """Get a thread-local session"""
    if not hasattr(thread_local, "session"):
        thread_local.session = Session()
    return thread_local.session

def close_session():
    """Close the thread-local session"""
    if hasattr(thread_local, "session"):
        thread_local.session.close()
        del thread_local.session

def create_user(username, password, email, is_superadmin=False, created_by=None):
    session = get_session()
    try:
        # Check if username already exists
        existing_user = session.query(User).filter_by(username=username).first()
        if existing_user:
            return False, "Username already exists"
        
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user = User(
            username=username, 
            hashed_password=hashed.decode('utf-8'), 
            email=email,
            is_superadmin=is_superadmin,
            created_by=created_by
        )
        session.add(user)
        session.commit()
        
        # Return the user object (not dictionary)
        return True, user
    except Exception as e:
        session.rollback()
        return False, f"Error creating user: {str(e)}"

def verify_user(username, password):
    session = get_session()
    try:
        user = session.query(User).filter_by(username=username, is_active=True).first()
        if user and bcrypt.checkpw(password.encode('utf-8'), user.hashed_password.encode('utf-8')):
            # Update last login
            user.last_login = datetime.utcnow()
            session.commit()
            
            # Return the user object (not dictionary)
            return user
        return None
    except Exception as e:
        return None

def get_all_users(current_user_id):
    """Get all users (superadmin can see all, regular users see only themselves)"""
    session = get_session()
    try:
        current_user = session.query(User).filter_by(user_id=current_user_id).first()
        
        if current_user and current_user.is_superadmin:
            users = session.query(User).all()
        else:
            users = session.query(User).filter_by(user_id=current_user_id).all()
        
        return users
    except Exception as e:
        return []

def reset_user_password(user_id, new_password, reset_by_superadmin=False):
    """Reset user password - can be done by superadmin or user themselves"""
    session = get_session()
    try:
        user = session.query(User).filter_by(user_id=user_id).first()
        if user:
            hashed = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
            user.hashed_password = hashed.decode('utf-8')
            session.commit()
            return True, "Password reset successfully"
        return False, "User not found"
    except Exception as e:
        session.rollback()
        return False, f"Error resetting password: {str(e)}"

def deactivate_user(user_id, deactivated_by):
    """Deactivate a user (superadmin only)"""
    session = get_session()
    try:
        deactivator = session.query(User).filter_by(user_id=deactivated_by).first()
        if not deactivator or not deactivator.is_superadmin:
            return False, "Only superadmin can deactivate users"
        
        user = session.query(User).filter_by(user_id=user_id).first()
        if user:
            user.is_active = False
            session.commit()
            return True, "User deactivated successfully"
        return False, "User not found"
    except Exception as e:
        session.rollback()
        return False, f"Error deactivating user: {str(e)}"

def activate_user(user_id, activated_by):
    """Activate a user (superadmin only)"""
    session = get_session()
    try:
        activator = session.query(User).filter_by(user_id=activated_by).first()
        if not activator or not activator.is_superadmin:
            return False, "Only superadmin can activate users"
        
        user = session.query(User).filter_by(user_id=user_id).first()
        if user:
            user.is_active = True
            session.commit()
            return True, "User activated successfully"
        return False, "User not found"
    except Exception as e:
        session.rollback()
        return False, f"Error activating user: {str(e)}"

def get_user_stats(user_id=None):
    """Get user statistics"""
    session = get_session()
    try:
        if user_id:
            # For specific user
            total_jobs = session.query(JobDescription).filter_by(user_id=user_id).count()
            total_candidates = session.query(Candidate).join(JobDescription).filter(
                JobDescription.user_id == user_id
            ).count()
        else:
            # For superadmin - all data
            total_jobs = session.query(JobDescription).count()
            total_candidates = session.query(Candidate).count()
        
        return total_jobs, total_candidates
    except Exception as e:
        return 0, 0

def get_recent_candidates(user_id=None, limit=10):
    """Get recent candidates"""
    session = get_session()
    try:
        if user_id:
            candidates = session.query(Candidate).join(JobDescription).filter(
                JobDescription.user_id == user_id
            ).order_by(Candidate.created_at.desc()).limit(limit).all()
        else:
            candidates = session.query(Candidate).order_by(Candidate.created_at.desc()).limit(limit).all()
        
        return candidates
    except Exception as e:
        return []

def store_cv_file(resume_code, filename, file_content, file_type, user_id):
    """Store CV file in database - handles duplicates by updating existing record"""
    session = get_session()
    try:
        # Check if CV already exists for this resume_code
        existing_cv = session.query(CVStorage).filter_by(resume_code=resume_code).first()
        
        if existing_cv:
            # Update existing record
            existing_cv.original_filename = filename
            existing_cv.file_content = file_content
            existing_cv.file_type = file_type
            existing_cv.uploaded_at = datetime.utcnow()
            existing_cv.user_id = user_id
            message = "CV updated successfully"
        else:
            # Create new record
            cv_record = CVStorage(
                resume_code=resume_code,
                original_filename=filename,
                file_content=file_content,
                file_type=file_type,
                user_id=user_id
            )
            session.add(cv_record)
            message = "CV stored successfully"
        
        session.commit()
        return True, message
    except Exception as e:
        session.rollback()
        return False, f"Error storing CV: {str(e)}"
    finally:
        close_session()

def get_cv_by_resume_code(resume_code, user_id):
    """Get CV file by resume code with access control"""
    session = get_session()
    try:
        cv_record = session.query(CVStorage).filter_by(resume_code=resume_code).first()
        if cv_record:
            # Check if user has access (owner or superadmin)
            user = session.query(User).filter_by(user_id=user_id).first()
            if user and (cv_record.user_id == user_id or user.is_superadmin):
                return cv_record
        return None
    except Exception as e:
        return None
    finally:
        close_session()

def get_cv_by_phone_number(phone_number, user_id):
    """Get CV by phone number (for confidential lookup)"""
    session = get_session()
    try:
        candidate = session.query(Candidate).filter_by(phone=phone_number).first()
        if candidate:
            cv_record = session.query(CVStorage).filter_by(resume_code=candidate.candidate_name.replace('Candidate-', '')).first()
            if cv_record:
                # Check if user has access
                user = session.query(User).filter_by(user_id=user_id).first()
                if user and (cv_record.user_id == user_id or user.is_superadmin):
                    return cv_record, candidate
        return None, None
    except Exception as e:
        return None, None
    finally:
        close_session()

def delete_job_analysis(job_id, user_id):
    """Delete job and all associated candidates, skills, education, experience, and CVs"""
    session = get_session()
    try:
        # Check if user has permission to delete this job
        job = session.query(JobDescription).filter_by(job_id=job_id).first()
        if not job:
            return False, "Job not found"
        
        user = session.query(User).filter_by(user_id=user_id).first()
        if not user:
            return False, "User not found"
        
        # Check permission (owner or superadmin)
        if not (user.is_superadmin or job.user_id == user_id):
            return False, "You don't have permission to delete this job"
        
        # Get all candidates for this job
        candidates = session.query(Candidate).filter_by(job_id=job_id).all()
        
        # Collect resume codes to delete CVs later
        resume_codes_to_delete = []
        
        # Delete all related data for each candidate
        for candidate in candidates:
            # Store resume code for CV deletion
            resume_code = candidate.candidate_name.replace('Candidate-', '')
            resume_codes_to_delete.append(resume_code)
            
            # Delete skills, education, and experience
            session.query(Skill).filter_by(candidate_id=candidate.candidate_id).delete()
            session.query(Education).filter_by(candidate_id=candidate.candidate_id).delete()
            session.query(WorkExperience).filter_by(candidate_id=candidate.candidate_id).delete()
            
            # Delete candidate
            session.delete(candidate)
        
        # Delete CV files that are only associated with this job's candidates
        for resume_code in resume_codes_to_delete:
            # Check if this resume_code is used by any other candidate
            other_candidate = session.query(Candidate).filter(
                Candidate.candidate_name.like(f'Candidate-{resume_code}%')
            ).first()
            
            if not other_candidate:
                # No other candidate uses this CV, so delete it
                cv_record = session.query(CVStorage).filter_by(resume_code=resume_code).first()
                if cv_record:
                    session.delete(cv_record)
        
        # Finally delete the job
        session.delete(job)
        session.commit()
        
        return True, f"Successfully deleted job '{job.job_title}' and all associated analysis data"
        
    except Exception as e:
        session.rollback()
        return False, f"Error deleting job analysis: {str(e)}"
    finally:
        close_session()

def check_duplicate_candidate(job_id, resume_code):
    """Check if a candidate with the same resume code already exists for this job"""
    session = get_session()
    try:
        candidate_name = f"Candidate-{resume_code}"
        existing_candidate = session.query(Candidate).filter_by(
            job_id=job_id, 
            candidate_name=candidate_name
        ).first()
        
        return existing_candidate is not None
    except Exception as e:
        return False
    finally:
        close_session()

# Create default superadmin if not exists
def create_default_superadmin():
    session = get_session()
    try:
        superadmin = session.query(User).filter_by(is_superadmin=True).first()
        if not superadmin:
            hashed = bcrypt.hashpw("admin123".encode('utf-8'), bcrypt.gensalt())
            superadmin = User(
                username="superadmin",
                hashed_password=hashed.decode('utf-8'),
                email="superadmin@resumeanalyzer.com",
                is_superadmin=True
            )
            session.add(superadmin)
            session.commit()
            print("Default superadmin created: username='superadmin', password='admin123'")
    except Exception as e:
        print(f"Error creating default superadmin: {e}")
    finally:
        close_session()

# Call this function when the module is imported
create_default_superadmin()
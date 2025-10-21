import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import re
import os

class AdvancedScoringModel:
    def __init__(self):
        self.skill_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.experience_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def load_model(self, filepath):
        """Load a trained model"""
        try:
            if os.path.exists(filepath):
                model_data = joblib.load(filepath)
                self.skill_vectorizer = model_data.get('skill_vectorizer', self.skill_vectorizer)
                self.experience_model = model_data.get('experience_model', self.experience_model)
                self.is_trained = True
                print("✅ Model loaded successfully!")
                return True
            else:
                print("⚠️ Model file not found. Using advanced rule-based scoring.")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("⚠️ Using advanced rule-based scoring instead.")
            return False
    
    def save_model(self, filepath):
        """Save the trained model"""
        try:
            model_data = {
                'skill_vectorizer': self.skill_vectorizer,
                'experience_model': self.experience_model
            }
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(model_data, filepath)
            print(f"✅ Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def analyze_candidate_fit(self, resume_data: dict, job_description: str) -> dict:
        """Advanced candidate-job fit analysis with comprehensive scoring"""
        
        # Comprehensive skill matching
        skill_fit = self._analyze_skill_fit_advanced(resume_data['skills'], job_description)
        
        # Experience suitability
        experience_fit = self._analyze_experience_fit_advanced(resume_data['experience'], job_description)
        
        # Qualification alignment
        qualification_fit = self._analyze_qualification_fit_advanced(resume_data['qualifications'], job_description)
        
        # Cultural and industry fit
        cultural_fit = self._analyze_cultural_fit_advanced(resume_data, job_description)
        
        # Awards and achievements impact
        achievements_impact = self._analyze_achievements_impact(resume_data['awards_certifications'])
        
        # Overall score calculation
        overall_score = self._calculate_comprehensive_score(
            skill_fit, experience_fit, qualification_fit, cultural_fit, achievements_impact
        )
        
        return {
            'overall_score': overall_score,
            'skill_analysis': skill_fit,
            'experience_analysis': experience_fit,
            'qualification_analysis': qualification_fit,
            'cultural_fit': cultural_fit,
            'achievements_impact': achievements_impact,
            'hiring_recommendation': self._generate_detailed_recommendation(overall_score, skill_fit, experience_fit),
            'interview_questions': self._generate_targeted_questions(resume_data, job_description),
            'strengths_weaknesses': self._identify_strengths_weaknesses(resume_data, job_description)
        }
    
    def generate_comprehensive_report(self, job_requirements: str, resume_data: dict) -> dict:
        """Generate comprehensive analysis report focusing on skills and experience"""
        
        # Perform advanced job fit analysis
        job_fit_analysis = self.analyze_candidate_fit(resume_data, job_requirements)
        
        # Generate quality analysis
        quality_analysis = self._generate_quality_analysis_advanced(resume_data)
        
        # Determine interview priority
        interview_priority = self._determine_interview_priority_advanced(job_fit_analysis['overall_score'])
        
        # Generate detailed recommendation
        recommendation = self._generate_comprehensive_recommendation_advanced(job_fit_analysis, quality_analysis)
        
        return {
            'job_fit_analysis': {
                'compatibility_score': job_fit_analysis['overall_score'],
                'skill_match': job_fit_analysis['skill_analysis']['match_percentage'],
                'experience_match': job_fit_analysis['experience_analysis']['experience_sufficiency_score'],
                'education_match': job_fit_analysis['qualification_analysis']['qualification_score'],
                'matching_skills': job_fit_analysis['skill_analysis']['exact_matches'],
                'missing_skills': job_fit_analysis['skill_analysis']['critical_skill_gaps'],
                'skill_match_breakdown': job_fit_analysis['skill_analysis']['category_breakdown'],
                'fit_analysis': job_fit_analysis['hiring_recommendation']
            },
            'quality_analysis': quality_analysis,
            'interview_priority': interview_priority,
            'recommendation': recommendation,
            'interview_questions': job_fit_analysis['interview_questions'],
            'strengths_weaknesses': job_fit_analysis['strengths_weaknesses'],
            'resume_code': resume_data.get('resume_code', 'N/A'),
            'confidential_lookup': f"Use phone number with resume code: {resume_data.get('resume_code', 'N/A')}"
        }
    
    def _analyze_skill_fit_advanced(self, skills_data: dict, job_description: str) -> dict:
        """Advanced skill fit analysis with category breakdown"""
        job_skills = self._extract_skills_with_context_advanced(job_description)
        candidate_skills_flat = self._flatten_skills_advanced(skills_data)
        
        # Calculate multiple skill metrics
        exact_matches = set(candidate_skills_flat) & set(job_skills)
        related_matches = self._find_related_skills_advanced(candidate_skills_flat, job_skills)
        skill_gaps = set(job_skills) - set(candidate_skills_flat)
        
        match_percentage = len(exact_matches) / len(job_skills) * 100 if job_skills else 0
        comprehensive_match = (len(exact_matches) + len(related_matches) * 0.7) / len(job_skills) * 100 if job_skills else 0
        
        # Category-based analysis
        category_breakdown = self._analyze_skill_categories(skills_data, job_skills)
        
        return {
            'match_percentage': round(match_percentage, 2),
            'comprehensive_match_percentage': round(comprehensive_match, 2),
            'exact_matches': list(exact_matches),
            'related_skills': list(related_matches),
            'critical_skill_gaps': list(skill_gaps),
            'skill_diversity_score': skills_data.get('skill_diversity', 0),
            'total_skills_count': skills_data.get('total_skills', 0),
            'category_breakdown': category_breakdown,
            'skill_strength_index': self._calculate_skill_strength_index(skills_data)
        }
    
    def _extract_skills_with_context_advanced(self, text: str) -> list[str]:
        """Extract skills with advanced context awareness"""
        skills = []
        
        # Comprehensive skill categories from nlp_processor
        skill_categories = {
            'programming_languages': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'swift', 'kotlin', 'php', 'ruby', 'scala', 'r', 'matlab', 'perl', 'html', 'css', 'sass', 'less', 'sql'],
            'web_development': ['django', 'flask', 'fastapi', 'spring', 'express', 'react', 'angular', 'vue', 'svelte', 'laravel', 'ruby on rails', 'asp.net', 'node.js', 'next.js', 'nuxt.js'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle', 'sql server', 'cassandra', 'dynamodb', 'elasticsearch'],
            'cloud_platforms': ['aws', 'azure', 'google cloud', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins'],
            'data_science_ai': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 'tableau', 'power bi'],
            'project_management': ['project management', 'agile', 'scrum', 'kanban', 'waterfall', 'prince2', 'pmp', 'jira', 'trello', 'asana'],
            'business_analysis': ['business analysis', 'requirements gathering', 'user stories', 'use cases', 'process modeling', 'bpmn', 'uml'],
            'finance_accounting': ['financial analysis', 'accounting', 'bookkeeping', 'financial modeling', 'budgeting', 'forecasting', 'quickbooks', 'xero'],
            'marketing_sales': ['digital marketing', 'social media marketing', 'seo', 'sem', 'google analytics', 'google ads', 'facebook ads', 'content marketing'],
            'soft_skills': ['communication', 'leadership', 'teamwork', 'problem solving', 'critical thinking', 'adaptability', 'time management'],
            'healthcare': ['patient care', 'medical terminology', 'healthcare management', 'clinical research'],
            'engineering': ['mechanical engineering', 'electrical engineering', 'civil engineering', 'chemical engineering'],
            'manufacturing': ['lean manufacturing', 'six sigma', 'quality assurance', 'production planning']
        }
        
        text_lower = text.lower()
        for category, skills_list in skill_categories.items():
            for skill in skills_list:
                if skill in text_lower:
                    skills.append(skill)
        
        return list(set(skills))
    
    def _flatten_skills_advanced(self, skills_data: dict) -> list[str]:
        """Flatten skills from all categories with confidence filtering"""
        all_skills = []
        skills_by_category = skills_data.get('skills_by_category', {})
        
        for category, skills in skills_by_category.items():
            for skill_info in skills:
                if isinstance(skill_info, dict) and 'skill' in skill_info:
                    # Only include skills with reasonable confidence
                    if skill_info.get('confidence', 0) > 0.3:
                        all_skills.append(skill_info['skill'])
        
        return all_skills
    
    def _find_related_skills_advanced(self, candidate_skills: list[str], job_skills: list[str]) -> set:
        """Find related skills with expanded relationships"""
        skill_relationships = {
            'python': ['django', 'flask', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'keras'],
            'java': ['spring', 'hibernate', 'j2ee', 'android', 'microservices'],
            'javascript': ['react', 'angular', 'vue', 'node.js', 'typescript', 'express'],
            'sql': ['mysql', 'postgresql', 'oracle', 'sql server', 'database design'],
            'aws': ['azure', 'google cloud', 'docker', 'kubernetes', 'terraform', 'devops'],
            'machine learning': ['deep learning', 'tensorflow', 'pytorch', 'neural networks', 'data science'],
            'react': ['react native', 'redux', 'next.js', 'frontend development'],
            'docker': ['kubernetes', 'containerization', 'microservices', 'devops'],
            'project management': ['agile', 'scrum', 'kanban', 'jira', 'stakeholder management'],
            'financial analysis': ['accounting', 'forecasting', 'budgeting', 'financial modeling']
        }
        
        related = set()
        for job_skill in job_skills:
            if job_skill in skill_relationships:
                for related_skill in skill_relationships[job_skill]:
                    if related_skill in candidate_skills:
                        related.add(related_skill)
        
        return related
    
    def _analyze_skill_categories(self, skills_data: dict, job_skills: list[str]) -> dict:
        """Analyze skill match by category"""
        category_analysis = {}
        skills_by_category = skills_data.get('skills_by_category', {})
        
        for category, skills in skills_by_category.items():
            category_skills = [s['skill'] for s in skills]
            matches = set(category_skills) & set(job_skills)
            category_analysis[category] = {
                'match_count': len(matches),
                'total_skills': len(category_skills),
                'match_percentage': len(matches) / len(job_skills) * 100 if job_skills else 0,
                'matched_skills': list(matches)
            }
        
        return category_analysis
    
    def _calculate_skill_strength_index(self, skills_data: dict) -> float:
        """Calculate overall skill strength index"""
        total_score = 0
        total_skills = 0
        
        for category, skills in skills_data.get('skills_by_category', {}).items():
            for skill in skills:
                confidence = skill.get('confidence', 0)
                frequency = skill.get('frequency', 1)
                total_score += confidence * frequency
                total_skills += 1
        
        return round((total_score / max(total_skills, 1)) * 100, 2) if total_skills > 0 else 0
    
    def _analyze_experience_fit_advanced(self, experience_data: dict, job_description: str) -> dict:
        """Advanced experience fit analysis"""
        # Extract experience requirements from job description
        exp_requirements = self._extract_experience_requirements_advanced(job_description)
        
        # Analyze career progression
        career_trajectory = self._analyze_career_trajectory_advanced(experience_data)
        
        # Industry relevance
        industry_relevance = self._analyze_industry_relevance_advanced(experience_data, job_description)
        
        # Leadership and responsibility
        leadership_score = self._assess_leadership_experience(experience_data)
        
        return {
            'experience_sufficiency': experience_data.get('total_experience_years', 0) >= exp_requirements.get('min_years', 0),
            'experience_sufficiency_score': self._calculate_experience_sufficiency_score(experience_data, exp_requirements),
            'career_progression_score': career_trajectory,
            'industry_relevance': industry_relevance,
            'leadership_experience': leadership_score,
            'achievement_density': len(experience_data.get('key_achievements', [])) / max(1, experience_data.get('total_experience_years', 1)),
            'career_stability': self._assess_career_stability(experience_data),
            'required_experience': exp_requirements.get('min_years', 0),
            'actual_experience': experience_data.get('total_experience_years', 0)
        }
    
    def _extract_experience_requirements_advanced(self, job_description: str) -> dict:
        """Extract detailed experience requirements with level detection"""
        patterns = {
            'min_years': r'(\d+)\+?\s*years?\s*experience',
            'max_years': r'(\d+)\s*-\s*(\d+)\s*years',
            'level': r'(senior|junior|mid-level|entry-level|executive|lead|principal)',
            'management': r'(management|manager|leadership|director|head of)'
        }
        
        requirements = {'min_years': 0, 'level': '', 'management_required': False}
        
        for key, pattern in patterns.items():
            match = re.search(pattern, job_description.lower())
            if match:
                if key == 'min_years':
                    requirements[key] = int(match.group(1))
                elif key == 'level':
                    requirements[key] = match.group(1)
                elif key == 'management':
                    requirements['management_required'] = True
        
        # Set default experience based on level
        if not requirements['min_years']:
            level_mapping = {'entry-level': 0, 'junior': 1, 'mid-level': 3, 'senior': 5, 'lead': 7, 'principal': 8, 'executive': 10}
            requirements['min_years'] = level_mapping.get(requirements['level'], 2)
        
        return requirements
    
    def _calculate_experience_sufficiency_score(self, experience_data: dict, requirements: dict) -> float:
        """Calculate experience sufficiency score"""
        actual_experience = experience_data.get('total_experience_years', 0)
        required_experience = requirements.get('min_years', 0)
        
        if actual_experience >= required_experience:
            return 100.0
        elif required_experience > 0:
            # Partial credit for having some experience
            return min(100.0, (actual_experience / required_experience) * 100)
        else:
            return 50.0  # Neutral score when no requirement specified
    
    def _analyze_career_trajectory_advanced(self, experience_data: dict) -> float:
        """Analyze career progression quality"""
        career_progression = experience_data.get('career_progression', [])
        if not career_progression:
            return 0.0
        
        # Multiple factors for career trajectory
        position_count = len(career_progression)
        total_experience = experience_data.get('total_experience_years', 0)
        avg_position_duration = total_experience / position_count if position_count > 0 else 0
        
        # Score based on stability and progression
        stability_score = min(1.0, avg_position_duration / 2.5) * 40  # Max 40 points
        progression_score = min(1.0, position_count / 8) * 30  # Max 30 points
        leadership_score = 30 if experience_data.get('promotion_trajectory') == 'Stable Progression' else 15
        
        return round(stability_score + progression_score + leadership_score, 2)
    
    def _analyze_industry_relevance_advanced(self, experience_data: dict, job_description: str) -> float:
        """Analyze industry experience relevance with weighted scoring"""
        job_industries = self._extract_industries_from_job_advanced(job_description)
        candidate_industries = experience_data.get('industry_sectors', [])
        
        if not job_industries or not candidate_industries:
            return 50.0
        
        # Weight recent industry experience more heavily
        overlap = set(job_industries) & set(candidate_industries)
        base_relevance = len(overlap) / len(job_industries) * 100
        
        # Bonus for multiple industry experiences
        industry_diversity_bonus = min(20, len(candidate_industries) * 5)
        
        return round(min(base_relevance + industry_diversity_bonus, 100), 2)
    
    def _extract_industries_from_job_advanced(self, job_description: str) -> list[str]:
        """Extract mentioned industries from job description"""
        industries = []
        text_lower = job_description.lower()
        
        industry_keywords = {
            'technology': ['software', 'it', 'tech', 'computer', 'developer', 'programming', 'technology'],
            'finance': ['banking', 'finance', 'investment', 'financial', 'accounting', 'bank', 'investment'],
            'healthcare': ['medical', 'health', 'hospital', 'healthcare', 'pharmaceutical', 'clinical'],
            'education': ['education', 'academic', 'university', 'school', 'teaching', 'educational'],
            'retail': ['retail', 'sales', 'customer service', 'merchandising', 'e-commerce'],
            'manufacturing': ['manufacturing', 'production', 'factory', 'industrial', 'engineering'],
            'consulting': ['consulting', 'consultant', 'advisory', 'professional services'],
            'government': ['government', 'public sector', 'municipal', 'civil service']
        }
        
        for industry, keywords in industry_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                industries.append(industry)
        
        return industries
    
    def _assess_leadership_experience(self, experience_data: dict) -> float:
        """Assess leadership and management experience"""
        leadership_keywords = [
            'lead', 'manage', 'direct', 'supervise', 'head', 'chief', 'director',
            'manager', 'supervisor', 'team lead', 'project lead', 'oversee', 'mentor'
        ]
        
        career_progression = experience_data.get('career_progression', [])
        leadership_evidence = 0
        
        for position in career_progression:
            description = position.get('description', '').lower()
            # Check for leadership keywords
            if any(keyword in description for keyword in leadership_keywords):
                leadership_evidence += 1
            
            # Check position titles for leadership roles
            position_title = position.get('position', '').lower()
            if any(role in position_title for role in ['manager', 'director', 'head', 'lead', 'chief']):
                leadership_evidence += 2
        
        return min(100.0, leadership_evidence * 20)  # Scale to 100
    
    def _assess_career_stability(self, experience_data: dict) -> float:
        """Assess career stability based on position duration and gaps"""
        career_progression = experience_data.get('career_progression', [])
        if not career_progression:
            return 50.0
        
        total_positions = len(career_progression)
        short_term_positions = sum(1 for pos in career_progression if pos.get('duration_years', 0) < 1.0)
        
        # Calculate stability score
        stability_ratio = (total_positions - short_term_positions) / total_positions
        return round(stability_ratio * 100, 2)
    
    def _analyze_qualification_fit_advanced(self, qualifications: dict, job_description: str) -> dict:
        """Advanced qualification fit analysis"""
        required_level = self._extract_required_qualification_level_advanced(job_description)
        candidate_level = qualifications.get('qualification_level', '')
        
        level_scores = {'phd': 100, 'masters': 85, 'bachelors': 70, 'diploma': 60, 'certificate': 50, '': 30}
        
        candidate_score = level_scores.get(candidate_level, 30)
        required_score = level_scores.get(required_level, 50)
        
        qualification_gap = max(0, required_score - candidate_score)
        qualification_sufficiency = candidate_score >= required_score
        
        return {
            'qualification_sufficiency': qualification_sufficiency,
            'qualification_gap': qualification_gap,
            'qualification_score': candidate_score,
            'required_level_score': required_score,
            'field_relevance': self._analyze_field_relevance_advanced(qualifications, job_description),
            'candidate_level': candidate_level,
            'required_level': required_level
        }
    
    def _extract_required_qualification_level_advanced(self, job_description: str) -> str:
        """Extract required qualification level from job description"""
        text_lower = job_description.lower()
        
        qualification_keywords = {
            'phd': ['phd', 'doctorate', 'doctoral'],
            'masters': ['master', 'masters', 'mba', 'msc', 'ma', 'postgraduate'],
            'bachelors': ['bachelor', 'degree', 'undergraduate', 'bsc', 'ba', 'bcom'],
            'diploma': ['diploma', 'certificate', 'certification']
        }
        
        for level, keywords in qualification_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return level
        
        return ''
    
    def _analyze_field_relevance_advanced(self, qualifications: dict, job_description: str) -> float:
        """Analyze relevance of qualification fields to job"""
        job_fields = self._extract_field_requirements_advanced(job_description)
        candidate_fields = qualifications.get('fields_of_study', [])
        
        if not job_fields or not candidate_fields:
            return 50.0
        
        overlap = set(job_fields) & set(candidate_fields)
        relevance = len(overlap) / len(job_fields) * 100
        
        # Bonus for multiple relevant fields
        if len(candidate_fields) > 1:
            relevance += min(20, (len(candidate_fields) - 1) * 5)
        
        return round(min(relevance, 100), 2)
    
    def _extract_field_requirements_advanced(self, job_description: str) -> list[str]:
        """Extract required fields of study from job description"""
        fields = []
        text_lower = job_description.lower()
        
        field_keywords = {
            'computer science': ['computer science', 'cs', 'software engineering', 'information systems', 'computing'],
            'business': ['business administration', 'mba', 'commerce', 'business', 'management'],
            'engineering': ['engineering', 'engineer', 'mechanical', 'electrical', 'civil', 'chemical'],
            'mathematics': ['mathematics', 'math', 'statistics', 'applied math'],
            'finance': ['finance', 'accounting', 'economics', 'banking', 'investment']
        }
        
        for field, keywords in field_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                fields.append(field)
        
        return fields
    
    def _analyze_cultural_fit_advanced(self, resume_data: dict, job_description: str) -> dict:
        """Analyze cultural and organizational fit"""
        # Extract company culture from job description
        culture_indicators = self._extract_culture_indicators_advanced(job_description)
        
        # Analyze candidate's adaptability based on career history
        adaptability_score = self._calculate_adaptability_score_advanced(resume_data)
        
        return {
            'culture_alignment': len(culture_indicators) * 10,  # Scale based on indicators
            'adaptability_score': adaptability_score,
            'potential_longevity': self._assess_longevity_potential_advanced(resume_data),
            'team_fit': self._assess_team_fit(resume_data, job_description)
        }
    
    def _extract_culture_indicators_advanced(self, job_description: str) -> list[str]:
        """Extract cultural indicators from job description"""
        indicators = []
        text_lower = job_description.lower()
        
        culture_keywords = [
            'fast-paced', 'innovative', 'collaborative', 'team-oriented', 'startup',
            'dynamic', 'creative', 'agile', 'flexible', 'remote', 'hybrid',
            'growth mindset', 'customer-focused', 'results-driven', 'entrepreneurial'
        ]
        
        for keyword in culture_keywords:
            if keyword in text_lower:
                indicators.append(keyword)
        
        return indicators
    
    def _calculate_adaptability_score_advanced(self, resume_data: dict) -> float:
        """Calculate candidate adaptability score"""
        score = 50.0  # Base score
        
        # Bonus for multiple industries
        industry_count = len(resume_data.get('experience', {}).get('industry_sectors', []))
        score += min(20, industry_count * 5)
        
        # Bonus for skill diversity
        skill_diversity = resume_data.get('skills', {}).get('skill_diversity', 0)
        score += min(20, skill_diversity * 2)
        
        # Bonus for career progression
        if resume_data.get('experience', {}).get('career_progression', []):
            score += 10
        
        # Bonus for multiple companies
        companies_count = len(resume_data.get('experience', {}).get('companies_worked', []))
        score += min(10, companies_count * 2)
        
        return min(score, 100.0)
    
    def _assess_longevity_potential_advanced(self, resume_data: dict) -> float:
        """Assess potential for long-term employment"""
        experience = resume_data.get('experience', {})
        total_experience = experience.get('total_experience_years', 0)
        
        if total_experience < 2:
            return 40.0  # Moderate risk for junior candidates
        
        # Calculate average position duration
        career_progression = experience.get('career_progression', [])
        if career_progression:
            avg_duration = total_experience / len(career_progression)
            if avg_duration >= 3.0:
                return 90.0  # Excellent stability
            elif avg_duration >= 2.0:
                return 75.0  # Good stability
            elif avg_duration >= 1.0:
                return 60.0  # Moderate stability
            else:
                return 45.0  # Lower stability
        
        return 50.0  # Neutral
    
    def _assess_team_fit(self, resume_data: dict, job_description: str) -> float:
        """Assess team fit based on skills and experience alignment"""
        # Simple team fit assessment
        skill_match = self._analyze_skill_fit_advanced(resume_data['skills'], job_description)
        experience_match = self._analyze_experience_fit_advanced(resume_data['experience'], job_description)
        
        return (skill_match['comprehensive_match_percentage'] + experience_match['experience_sufficiency_score']) / 2
    
    def _analyze_achievements_impact(self, awards_data: dict) -> dict:
        """Analyze impact of awards and achievements"""
        total_achievements = (
            len(awards_data.get('awards', [])) +
            len(awards_data.get('certifications', [])) +
            len(awards_data.get('honors', [])) +
            len(awards_data.get('publications', [])) +
            len(awards_data.get('patents', []))
        )
        
        achievement_score = min(100, total_achievements * 15)  # Scale achievements
        
        return {
            'achievement_score': achievement_score,
            'total_achievements': total_achievements,
            'has_awards': len(awards_data.get('awards', [])) > 0,
            'has_certifications': len(awards_data.get('certifications', [])) > 0,
            'has_publications': len(awards_data.get('publications', [])) > 0
        }
    
    def _calculate_comprehensive_score(self, skill_fit: dict, experience_fit: dict, 
                                    qualification_fit: dict, cultural_fit: dict, 
                                    achievements_impact: dict) -> float:
        """Calculate overall candidate score with weighted factors"""
        weights = {
            'skill_match': 0.40,  # Highest weight for skills
            'experience': 0.30,   # Strong emphasis on experience
            'qualifications': 0.15, # Moderate weight for qualifications
            'cultural_fit': 0.10, # Cultural fit
            'achievements': 0.05   # Bonus for achievements
        }
        
        skill_score = skill_fit.get('comprehensive_match_percentage', 0)
        experience_score = experience_fit.get('experience_sufficiency_score', 0)
        qualification_score = qualification_fit.get('qualification_score', 0)
        cultural_score = (cultural_fit.get('culture_alignment', 0) + cultural_fit.get('adaptability_score', 0)) / 2
        achievements_score = achievements_impact.get('achievement_score', 0)
        
        total_score = (
            skill_score * weights['skill_match'] +
            experience_score * weights['experience'] +
            qualification_score * weights['qualifications'] +
            cultural_score * weights['cultural_fit'] +
            achievements_score * weights['achievements']
        )
        
        return round(total_score, 2)
    
    def _generate_detailed_recommendation(self, score: float, skill_fit: dict, experience_fit: dict) -> str:
        """Generate detailed hiring recommendation"""
        skill_match = skill_fit.get('comprehensive_match_percentage', 0)
        experience_sufficient = experience_fit.get('experience_sufficiency', False)
        
        if score >= 90:
            return "🏆 TOP CANDIDATE - Exceptional match with outstanding skills and experience. Immediate interview recommended."
        elif score >= 80:
            return "✅ STRONG RECOMMEND - Excellent skills alignment with sufficient experience. High potential candidate."
        elif score >= 70:
            return "👍 RECOMMEND - Good overall fit with solid qualifications. Worth interviewing."
        elif score >= 60:
            if skill_match >= 70:
                return "🟡 SKILL-FOCUSED CANDIDATE - Strong skills match but limited experience. Consider for specialized roles."
            else:
                return "🟡 EXPERIENCED CANDIDATE - Good experience but some skill gaps. Training potential."
        elif score >= 50:
            return "🟠 BORDERLINE CANDIDATE - Moderate fit with development areas. Consider for junior positions."
        else:
            return "🔴 NOT RECOMMENDED - Significant mismatches with role requirements."
    
    def _generate_targeted_questions(self, resume_data: dict, job_description: str) -> list[str]:
        """Generate targeted interview questions based on analysis"""
        questions = []
        
        # Skill gap questions
        skill_gaps = resume_data.get('skill_analysis', {}).get('critical_skill_gaps', [])
        if skill_gaps:
            questions.append(f"How would you approach developing skills in {', '.join(skill_gaps[:2])} to meet our requirements?")
        
        # Experience questions
        experience_data = resume_data.get('experience', {})
        if experience_data.get('total_experience_years', 0) < 3:
            questions.append("What strategies would you use to quickly ramp up in this role given your experience level?")
        
        # Career progression questions
        if len(experience_data.get('career_progression', [])) > 0:
            questions.append("Can you walk us through your career progression and what motivated your key transitions?")
        
        # Achievement questions
        if experience_data.get('key_achievements', []):
            questions.append("Which of your professional achievements are you most proud of and how do they relate to this role?")
        
        # Industry-specific questions
        industries = experience_data.get('industry_sectors', [])
        if industries:
            questions.append(f"How has your experience in {industries[0]} prepared you for this position?")
        
        # Add behavioral questions
        questions.extend([
            "Describe a challenging project you worked on and how you overcame obstacles.",
            "How do you stay current with industry trends and technologies?",
            "What attracted you to this particular role and our company?"
        ])
        
        return questions[:6]  # Return max 6 questions
    
    def _identify_strengths_weaknesses(self, resume_data: dict, job_description: str) -> dict:
        """Identify key strengths and weaknesses for the role"""
        strengths = []
        weaknesses = []
        
        # Skill-based strengths/weaknesses
        skill_analysis = self._analyze_skill_fit_advanced(resume_data['skills'], job_description)
        if skill_analysis['comprehensive_match_percentage'] >= 70:
            strengths.append(f"Strong skills alignment ({skill_analysis['comprehensive_match_percentage']}% match)")
        else:
            weaknesses.append(f"Skill gaps in: {', '.join(skill_analysis['critical_skill_gaps'][:3])}")
        
        # Experience-based strengths/weaknesses
        experience_data = resume_data.get('experience', {})
        if experience_data.get('total_experience_years', 0) >= 3:
            strengths.append(f"Substantial professional experience ({experience_data['total_experience_years']} years)")
        else:
            weaknesses.append("Limited professional experience")
        
        # Qualification strengths
        qualifications = resume_data.get('qualifications', {})
        if qualifications.get('qualification_level') in ['masters', 'phd']:
            strengths.append("Advanced educational qualifications")
        
        # Career stability
        if experience_data.get('promotion_trajectory') == 'Stable Progression':
            strengths.append("Stable career progression")
        
        return {
            'strengths': strengths[:4],
            'weaknesses': weaknesses[:3],
            'development_areas': skill_analysis['critical_skill_gaps'][:3]
        }
    
    def _generate_quality_analysis_advanced(self, resume_data: dict) -> dict:
        """Generate advanced resume quality analysis"""
        quality_score = 65  # Base score
        
        # Adjust based on content quality
        experience_years = resume_data.get('experience', {}).get('total_experience_years', 0)
        if experience_years > 2:
            quality_score += 10
        if experience_years > 5:
            quality_score += 5
        
        total_skills = resume_data.get('skills', {}).get('total_skills', 0)
        if total_skills > 10:
            quality_score += 10
        if total_skills > 20:
            quality_score += 5
        
        if resume_data.get('qualifications', {}).get('highest_qualification'):
            quality_score += 10
        
        # Achievements bonus
        awards = resume_data.get('awards_certifications', {})
        total_achievements = sum(len(awards.get(category, [])) for category in ['awards', 'certifications', 'honors'])
        if total_achievements > 0:
            quality_score += min(10, total_achievements * 2)
        
        # Cap at 100
        quality_score = min(quality_score, 100)
        
        return {
            'overall_score': quality_score,
            'section_scores': {
                'Experience': min(100, experience_years * 8 + 40),
                'Skills': min(100, total_skills * 3),
                'Education': 100 if resume_data.get('qualifications', {}).get('highest_qualification') else 50,
                'Achievements': min(100, total_achievements * 20)
            },
            'strengths': self._identify_resume_strengths(resume_data),
            'improvements': self._identify_resume_improvements(resume_data),
            'ats_compatibility': 80  # Good ATS compatibility
        }
    
    def _identify_resume_strengths(self, resume_data: dict) -> list:
        """Identify resume strengths"""
        strengths = []
        
        if resume_data.get('experience', {}).get('total_experience_years', 0) >= 3:
            strengths.append("Clear career progression and substantial experience")
        
        if resume_data.get('skills', {}).get('total_skills', 0) >= 15:
            strengths.append("Diverse and comprehensive skill set")
        
        if resume_data.get('qualifications', {}).get('highest_qualification'):
            strengths.append("Well-documented educational background")
        
        if resume_data.get('awards_certifications', {}).get('awards') or resume_data.get('awards_certifications', {}).get('certifications'):
            strengths.append("Notable professional achievements and certifications")
        
        return strengths
    
    def _identify_resume_improvements(self, resume_data: dict) -> list:
        """Identify resume improvement areas"""
        improvements = []
        
        if resume_data.get('experience', {}).get('total_experience_years', 0) < 2:
            improvements.append("Could benefit from more detailed project descriptions")
        
        if resume_data.get('skills', {}).get('total_skills', 0) < 8:
            improvements.append("Consider expanding technical and professional skills section")
        
        if not resume_data.get('qualifications', {}).get('highest_qualification'):
            improvements.append("Include educational qualifications if applicable")
        
        return improvements
    
    def _determine_interview_priority_advanced(self, score: float) -> str:
        """Determine interview priority based on comprehensive score"""
        if score >= 85:
            return "HIGH PRIORITY - Schedule immediately"
        elif score >= 70:
            return "MEDIUM PRIORITY - Schedule within week"
        elif score >= 55:
            return "LOW PRIORITY - Consider if other candidates unavailable"
        else:
            return "NOT RECOMMENDED - Significant gaps identified"
    
    def _generate_comprehensive_recommendation_advanced(self, job_fit: dict, quality: dict) -> str:
        """Generate comprehensive hiring recommendation"""
        overall_score = job_fit['overall_score']
        quality_score = quality['overall_score']
        
        if overall_score >= 85 and quality_score >= 80:
            return "🏆 EXCEPTIONAL CANDIDATE - Outstanding match across all criteria with high-quality resume"
        elif overall_score >= 75:
            return "✅ STRONG CANDIDATE - Excellent job fit with well-presented qualifications"
        elif overall_score >= 60:
            return "👍 PROMISING CANDIDATE - Good potential with some areas for development"
        elif overall_score >= 50:
            return "🟡 DEVELOPMENT CANDIDATE - Requires training but shows potential"
        else:
            return "🔴 NOT SUITABLE - Significant gaps in required qualifications"

# Create alias for backward compatibility
ScoringModel = AdvancedScoringModel
import pandas as pd
import numpy as np
from faker import Faker
import yaml
import os

def generate_synthetic_data(config_path='config/config.yaml'):
    """
    Generates a synthetic EdTech dataset with 5000 student records.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    fake = Faker()
    Faker.seed(42)
    np.random.seed(42)
    
    num_students = 5000
    
    # Basic info
    student_ids = [f"STU_{1000 + i}" for i in range(num_students)]
    ages = np.random.randint(18, 46, size=num_students)
    genders = np.random.choice(['M', 'F', 'Other'], size=num_students)
    regions = np.random.choice(['North', 'South', 'East', 'West', 'Central'], size=num_students)
    course_ids = [f"C{str(i).zfill(3)}" for i in np.random.randint(1, 11, size=num_students)]
    
    # Enrollment dates (within the last year)
    enrollment_dates = [fake.date_between(start_date='-1y', end_date='today') for _ in range(num_students)]
    
    # Engagement metrics
    total_logins = np.random.randint(1, 201, size=num_students)
    avg_session_duration = np.random.uniform(1, 121, size=num_students)
    modules_completed = np.random.randint(0, 21, size=num_students)
    total_modules = [20] * num_students
    
    # Quiz scores (avg of 5)
    quiz_avg = np.random.uniform(0, 101, size=num_students)
    
    forum_posts = np.random.randint(0, 51, size=num_students)
    assignment_submissions = np.random.randint(0, 11, size=num_students)
    video_watch_pct = np.random.uniform(0, 101, size=num_students)
    days_since_last_login = np.random.randint(0, 91, size=num_students)
    
    # Target: dropped_out (30% rate, correlated with low engagement)
    # Simple risk score for data generation
    norm_logins = total_logins / 200
    norm_duration = avg_session_duration / 120
    norm_completion = modules_completed / 20
    norm_quiz = quiz_avg / 100
    norm_last_login = 1 - (days_since_last_login / 90)
    
    engagement_proxy = (0.3 * norm_logins + 0.2 * norm_duration + 
                        0.2 * norm_completion + 0.2 * norm_quiz + 0.1 * norm_last_login)
    
    # Higher engagement_proxy -> lower dropout probability
    # Base dropout rate ~30%
    dropout_prob = 1 - engagement_proxy
    # Adjust to get ~30% mean
    dropout_prob = np.clip(dropout_prob * 0.6, 0, 1) 
    dropped_out = (np.random.rand(num_students) < dropout_prob).astype(int)
    
    # Verify dropout rate
    print(f"Generated dropout rate: {dropped_out.mean():.2%}")
    
    df = pd.DataFrame({
        'student_id': student_ids,
        'age': ages,
        'gender': genders,
        'region': regions,
        'course_id': course_ids,
        'enrollment_date': enrollment_dates,
        'total_logins': total_logins,
        'avg_session_duration_mins': avg_session_duration,
        'modules_completed': modules_completed,
        'total_modules': total_modules,
        'quiz_scores_avg': quiz_avg,
        'forum_posts': forum_posts,
        'assignment_submissions': assignment_submissions,
        'video_watch_pct': video_watch_pct,
        'days_since_last_login': days_since_last_login,
        'dropped_out': dropped_out
    })
    
    output_path = config['paths']['raw_data']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    generate_synthetic_data()

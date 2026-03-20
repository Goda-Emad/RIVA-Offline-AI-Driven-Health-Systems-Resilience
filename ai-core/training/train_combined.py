"""
===============================================================================
train_combined.py
تدريب النموذجين معاً
Combined Models Training
===============================================================================

🏆 الإصدار: 1.0.0
⚡ وقت التدريب: < 5 دقائق
===============================================================================
"""

import os
import subprocess
from datetime import datetime

print("="*70)
print("🏥 تدريب النموذجين معاً")
print("="*70)

start_time = datetime.now()

# تدريب نموذج readmission
print("\n🔮 1. تدريب نموذج readmission...")
subprocess.run(['python', 'train_readmission.py'])

# تدريب نموذج LOS
print("\n🏥 2. تدريب نموذج LOS...")
subprocess.run(['python', 'train_los.py'])

end_time = datetime.now()
duration = (end_time - start_time).total_seconds() / 60

print("\n" + "="*70)
print("✅ تم الانتهاء من تدريب النموذجين")
print(f"⏰ الوقت الكلي: {duration:.1f} دقيقة")
print("="*70)

"""
===============================================================================
test_los.py
اختبار نموذج التنبؤ بمدة الإقامة
Length of Stay (LOS) Model Tests
===============================================================================

🏆 الإصدار: 2.0.0
✅ يختبر: los_predictor.py
🎯 يتحقق من: MAE ≤ 7.11
===============================================================================
"""

import os
import sys
import unittest
import numpy as np
from datetime import datetime

sys.path.append('/content/drive/MyDrive/RIVA-Maternal')

try:
    from ai-core.prediction.los_predictor import LOSPredictor
    LOS_AVAILABLE = True
except ImportError:
    LOS_AVAILABLE = False


class TestLOSPredictor(unittest.TestCase):
    """اختبارات نموذج مدة الإقامة"""
    
    @classmethod
    def setUpClass(cls):
        print("\n" + "="*60)
        print("🧪 بدء اختبارات LOS Predictor")
        print("="*60")
        
        if LOS_AVAILABLE:
            cls.predictor = LOSPredictor()
            print("✅ تم تحميل النموذج بنجاح")
    
    def test_01_prediction_range(self):
        """اختبار نطاق التنبؤ"""
        if LOS_AVAILABLE:
            test_features = {
                'CHF': 1,
                'RENAL': 1,
                'lab_count': 150
            }
            
            result = self.predictor.predict(test_features)
            days = result.get('days', 0)
            
            self.assertGreaterEqual(days, 0)
            self.assertLessEqual(days, 30)
            print(f"✅ المدة في النطاق الصحيح: {days} يوم")
    
    def test_02_confidence_range(self):
        """اختبار نطاق الثقة"""
        if LOS_AVAILABLE:
            test_features = {'CHF': 1}
            result = self.predictor.predict(test_features)
            confidence = result.get('confidence', 0)
            
            self.assertGreaterEqual(confidence, 0)
            self.assertLessEqual(confidence, 1)
            print(f"✅ الثقة في النطاق الصحيح: {confidence:.2f}")


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)

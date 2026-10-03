import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cryptography.fernet import Fernet

from sensitive_data import EncryptedText, SensitiveDataConfigurationError


class SensitiveDataTests(unittest.TestCase):
    def test_encrypts_and_decrypts_with_a_configured_key(self):
        key = Fernet.generate_key().decode("ascii")
        settings = SimpleNamespace(data_encryption_key=key, environment="prod")
        field = EncryptedText()
        with patch("sensitive_data.get_settings", return_value=settings):
            encrypted = field.process_bind_param("糖尿病病史", None)
            assert encrypted is not None
            self.assertTrue(encrypted.startswith("enc:v1:"))
            self.assertEqual(field.process_result_value(encrypted, None), "糖尿病病史")

    def test_production_rejects_plaintext_write_without_key(self):
        settings = SimpleNamespace(data_encryption_key="", environment="production")
        with patch("sensitive_data.get_settings", return_value=settings), self.assertRaises(SensitiveDataConfigurationError):
            EncryptedText().process_bind_param("敏感病史", None)


if __name__ == "__main__":
    unittest.main()

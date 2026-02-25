"""
Unit tests for pages/_utils.py IIF generation functions.
"""

import unittest
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pages._utils import (
    get_qb_item_code,
    qb_date,
    validate_items_against_qb,
    generate_purchase_order_iif,
    generate_sales_order_iif,
)


class TestGetQbItemCode(unittest.TestCase):
    """Tests for get_qb_item_code function."""

    def test_returns_type_code_when_present(self):
        row = {"type_code": "IC-CB02-60", "techpia_code": "TP-123", "item": "Carbon", "vlt": 2}
        self.assertEqual(get_qb_item_code(row), "IC-CB02-60")

    def test_returns_techpia_code_when_no_type_code(self):
        row = {"type_code": "", "techpia_code": "TP-123", "item": "Carbon", "vlt": 2}
        self.assertEqual(get_qb_item_code(row), "TP-123")

    def test_returns_techpia_code_when_type_code_is_none(self):
        row = {"type_code": None, "techpia_code": "TP-456", "item": "Carbon", "vlt": 5}
        self.assertEqual(get_qb_item_code(row), "TP-456")

    def test_fallback_when_no_codes(self):
        row = {"type_code": "", "techpia_code": "", "item": "Carbon", "vlt": 2, "width": 60}
        result = get_qb_item_code(row)
        self.assertIn("Carbon", result)
        self.assertIn("2%", result)

    def test_fallback_with_missing_fields(self):
        row = {}
        result = get_qb_item_code(row)
        self.assertIn("Unknown", result)


class TestQbDate(unittest.TestCase):
    """Tests for qb_date function."""

    def test_formats_datetime_object(self):
        d = datetime(2024, 3, 15)
        self.assertEqual(qb_date(d), "03/15/2024")

    def test_returns_string_as_is(self):
        self.assertEqual(qb_date("12/25/2024"), "12/25/2024")

    def test_defaults_to_today(self):
        result = qb_date()
        today = datetime.today().strftime("%m/%d/%Y")
        self.assertEqual(result, today)


class TestValidateItemsAgainstQb(unittest.TestCase):
    """Tests for validate_items_against_qb function."""

    def test_returns_empty_when_all_items_exist(self):
        rows = [
            {"type_code": "IC-CB02-60", "description": "Carbon 2%"},
            {"type_code": "IC-CB05-40", "description": "Carbon 5%"},
        ]
        qb_items = {"IC-CB02-60", "IC-CB05-40", "IC-CB10-40"}
        missing = validate_items_against_qb(rows, qb_items)
        self.assertEqual(missing, [])

    def test_returns_missing_items(self):
        rows = [
            {"type_code": "IC-CB02-60", "description": "Carbon 2%"},
            {"type_code": "INVALID-CODE", "description": "Invalid Item"},
        ]
        qb_items = {"IC-CB02-60", "IC-CB05-40"}
        missing = validate_items_against_qb(rows, qb_items)
        self.assertEqual(len(missing), 1)
        self.assertIn("INVALID-CODE", missing[0])

    def test_handles_empty_rows(self):
        missing = validate_items_against_qb([], {"IC-CB02-60"})
        self.assertEqual(missing, [])


class TestGeneratePurchaseOrderIif(unittest.TestCase):
    """Tests for generate_purchase_order_iif function."""

    def test_generates_valid_iif_structure(self):
        rows = [
            {"type_code": "IC-CB02-60", "quantity": 10, "po_unit_price": "25.00", "po_amount": "250.00"},
        ]
        result = generate_purchase_order_iif(rows, vendor_name="TestVendor", txn_date="02/09/2026")
        
        lines = result.split("\n")
        
        # Check headers
        self.assertTrue(lines[0].startswith("!TRNS"))
        self.assertTrue(lines[1].startswith("!SPL"))
        self.assertTrue(lines[2].startswith("!ENDTRNS"))
        
        # Check TRNS line
        self.assertIn("TRNS\tPURCHORD", result)
        self.assertIn("TestVendor", result)
        self.assertIn("02/09/2026", result)
        
        # Check SPL line
        self.assertIn("SPL\tPURCHORD", result)
        self.assertIn("IC-CB02-60", result)
        
        # Check ends with ENDTRNS
        self.assertTrue(lines[-1] == "ENDTRNS")

    def test_multiple_items_in_single_transaction(self):
        rows = [
            {"type_code": "IC-CB02-60", "quantity": 10, "po_unit_price": "25.00", "po_amount": "250.00"},
            {"type_code": "IC-CB05-40", "quantity": 5, "po_unit_price": "30.00", "po_amount": "150.00"},
        ]
        result = generate_purchase_order_iif(rows, vendor_name="TestVendor", txn_date="02/09/2026")
        
        lines = result.split("\n")
        
        # Should have: 3 headers + 1 TRNS + 2 SPL + 1 ENDTRNS = 7 lines
        self.assertEqual(len(lines), 7)
        
        # Count SPL lines
        spl_lines = [l for l in lines if l.startswith("SPL")]
        self.assertEqual(len(spl_lines), 2)
        
        # Only one TRNS line (grouped transaction)
        trns_lines = [l for l in lines if l.startswith("TRNS\t")]
        self.assertEqual(len(trns_lines), 1)

    def test_handles_formatted_numbers(self):
        rows = [
            {"type_code": "IC-CB02-60", "quantity": "1,000", "po_unit_price": "1,234.56", "po_amount": "1,234,560.00"},
        ]
        result = generate_purchase_order_iif(rows, vendor_name="TestVendor")
        
        # Should parse numbers correctly (commas removed)
        self.assertIn("1000.0", result)  # quantity
        self.assertIn("1234.56", result)  # price

    def test_uses_inventory_asset_account(self):
        rows = [{"type_code": "IC-CB02-60", "quantity": 10, "po_unit_price": "25.00", "po_amount": "250.00"}]
        result = generate_purchase_order_iif(rows, vendor_name="TestVendor")
        
        self.assertIn("Inventory Asset", result)

    def test_custom_docnum(self):
        rows = [{"type_code": "IC-CB02-60", "quantity": 10, "po_unit_price": "25.00", "po_amount": "250.00"}]
        result = generate_purchase_order_iif(rows, vendor_name="TestVendor", docnum=99999)
        
        self.assertIn("99999", result)


class TestGenerateSalesOrderIif(unittest.TestCase):
    """Tests for generate_sales_order_iif function."""

    def test_generates_valid_iif_structure(self):
        rows = [
            {"type_code": "IC-CB02-60", "quantity": 10, "pi_unit_price": "35.00", "pi_amount": "350.00"},
        ]
        result = generate_sales_order_iif(rows, customer_name="TestCustomer", txn_date="02/09/2026")
        
        lines = result.split("\n")
        
        # Check headers
        self.assertTrue(lines[0].startswith("!TRNS"))
        self.assertTrue(lines[1].startswith("!SPL"))
        self.assertTrue(lines[2].startswith("!ENDTRNS"))
        
        # Check TRNS line
        self.assertIn("TRNS\tSALESORD", result)
        self.assertIn("TestCustomer", result)
        self.assertIn("02/09/2026", result)
        
        # Check SPL line
        self.assertIn("SPL\tSALESORD", result)
        self.assertIn("IC-CB02-60", result)
        
        # Check ends with ENDTRNS
        self.assertTrue(lines[-1] == "ENDTRNS")

    def test_multiple_items_in_single_transaction(self):
        rows = [
            {"type_code": "IC-CB02-60", "quantity": 10, "pi_unit_price": "35.00", "pi_amount": "350.00"},
            {"type_code": "IC-CB05-40", "quantity": 5, "pi_unit_price": "40.00", "pi_amount": "200.00"},
        ]
        result = generate_sales_order_iif(rows, customer_name="TestCustomer", txn_date="02/09/2026")
        
        lines = result.split("\n")
        
        # Should have: 3 headers + 1 TRNS + 2 SPL + 1 ENDTRNS = 7 lines
        self.assertEqual(len(lines), 7)
        
        # Count SPL lines
        spl_lines = [l for l in lines if l.startswith("SPL")]
        self.assertEqual(len(spl_lines), 2)

    def test_uses_sales_account(self):
        rows = [{"type_code": "IC-CB02-60", "quantity": 10, "pi_unit_price": "35.00", "pi_amount": "350.00"}]
        result = generate_sales_order_iif(rows, customer_name="TestCustomer")
        
        # SPL lines should use Sales account
        self.assertIn("\tSales\t", result)

    def test_uses_accounts_receivable(self):
        rows = [{"type_code": "IC-CB02-60", "quantity": 10, "pi_unit_price": "35.00", "pi_amount": "350.00"}]
        result = generate_sales_order_iif(rows, customer_name="TestCustomer")
        
        # TRNS line should use Accounts Receivable
        self.assertIn("Accounts Receivable", result)


class TestIifIntegration(unittest.TestCase):
    """Integration tests for IIF generation."""

    def test_po_and_so_have_same_structure(self):
        """Both PO and SO should follow the same IIF structure pattern."""
        rows = [{"type_code": "IC-CB02-60", "quantity": 10, "po_unit_price": "25.00", "po_amount": "250.00", 
                 "pi_unit_price": "35.00", "pi_amount": "350.00"}]
        
        po = generate_purchase_order_iif(rows, vendor_name="Vendor", txn_date="01/01/2026")
        so = generate_sales_order_iif(rows, customer_name="Customer", txn_date="01/01/2026")
        
        po_lines = po.split("\n")
        so_lines = so.split("\n")
        
        # Same number of lines
        self.assertEqual(len(po_lines), len(so_lines))
        
        # Same header structure
        self.assertEqual(po_lines[0], so_lines[0])
        self.assertEqual(po_lines[1], so_lines[1])
        self.assertEqual(po_lines[2], so_lines[2])

    def test_empty_rows_generates_minimal_iif(self):
        """Empty rows should still generate valid IIF with headers."""
        po = generate_purchase_order_iif([], vendor_name="Vendor")
        so = generate_sales_order_iif([], customer_name="Customer")
        
        # Should have headers + TRNS + ENDTRNS (no SPL lines)
        po_lines = po.split("\n")
        so_lines = so.split("\n")
        
        self.assertEqual(len(po_lines), 5)  # 3 headers + 1 TRNS + 1 ENDTRNS
        self.assertEqual(len(so_lines), 5)


if __name__ == "__main__":
    unittest.main()

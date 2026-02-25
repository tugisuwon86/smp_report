def load_qb_lists_from_iif(path):
    items = set()
    vendors = set()
    customers = set()

    with open(path, encoding="cp1252", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue

            rec = parts[0]

            if rec == "INVITEM":
                items.add(parts[1])

            elif rec == "VEND":
                vendors.add(parts[1])

            elif rec == "CUST":
                customers.add(parts[1])

    return items, vendors, customers


from datetime import datetime


def get_qb_item_code(row):
    """
    Get QuickBooks item code from row.
    Prioritizes type_code (from metadata), falls back to techpia_code.
    """
    if row.get("type_code"):
        return row["type_code"]
    if row.get("techpia_code"):
        return row["techpia_code"]
    # Fallback: construct a name (may not exist in QB)
    base = f"{row.get('item', 'Unknown')} {row.get('vlt', 0)}%"
    return f"{base} {row.get('width', 0)}\""


def qb_date(d=None):
    """Format date for QuickBooks IIF (MM/DD/YYYY)"""
    if d is None:
        d = datetime.today()
    if isinstance(d, str):
        return d
    return d.strftime("%m/%d/%Y")


def validate_items_against_qb(rows, qb_items):
    """
    Validate that all item codes in rows exist in QuickBooks.
    Returns list of missing items.
    """
    missing = []
    for r in rows:
        code = get_qb_item_code(r)
        if code and code not in qb_items:
            missing.append(f"{code} ({r.get('description', 'No description')})")
    return missing


def generate_purchase_order_iif(rows, vendor_name, txn_date=None, docnum=50001):
    """
    Generate IIF content for a Purchase Order.
    
    Args:
        rows: List of dicts with keys: type_code, quantity, po_unit_price, po_amount
        vendor_name: Name of the vendor (must exist in QB)
        txn_date: Transaction date (defaults to today)
        docnum: Document number for the PO
    
    Returns:
        String containing the IIF file content
    """
    if txn_date is None:
        txn_date = qb_date()
    
    lines = [
        "!TRNS\tTRNSTYPE\tDATE\tACCNT\tNAME\tDOCNUM",
        "!SPL\tTRNSTYPE\tDATE\tACCNT\tNAME\tQNTY\tPRICE\tAMOUNT\tITEM",
        "!ENDTRNS",
    ]
    
    # TRNS header line (one per PO)
    lines.append(
        f"TRNS\tPURCHORD\t{txn_date}\tAccounts Payable\t{vendor_name}\t{docnum}"
    )
    
    # SPL lines (one per item)
    for r in rows:
        item_code = get_qb_item_code(r)
        
        # Parse numeric values (handle formatted strings like "1,234.56")
        try:
            qty = float(str(r.get("quantity", 0)).replace(",", ""))
        except (ValueError, TypeError):
            qty = 0
        
        try:
            price = float(str(r.get("po_unit_price", 0)).replace(",", ""))
        except (ValueError, TypeError):
            price = 0
        
        try:
            amount = float(str(r.get("po_amount", 0)).replace(",", ""))
        except (ValueError, TypeError):
            amount = qty * price
        
        lines.append(
            f"SPL\tPURCHORD\t{txn_date}\tInventory Asset\t{vendor_name}\t{qty}\t{price}\t{amount}\t{item_code}"
        )
    
    lines.append("ENDTRNS")
    
    return "\n".join(lines)


def generate_sales_order_iif(rows, customer_name, txn_date=None, docnum=10001):
    """
    Generate IIF content for a Sales Order.
    
    Args:
        rows: List of dicts with keys: type_code, quantity, pi_unit_price, pi_amount
        customer_name: Name of the customer (must exist in QB)
        txn_date: Transaction date (defaults to today)
        docnum: Document number for the SO
    
    Returns:
        String containing the IIF file content
    """
    if txn_date is None:
        txn_date = qb_date()
    
    lines = [
        "!TRNS\tTRNSTYPE\tDATE\tACCNT\tNAME\tDOCNUM",
        "!SPL\tTRNSTYPE\tDATE\tACCNT\tNAME\tQNTY\tPRICE\tAMOUNT\tITEM",
        "!ENDTRNS",
    ]
    
    # TRNS header line (one per SO)
    lines.append(
        f"TRNS\tSALESORD\t{txn_date}\tAccounts Receivable\t{customer_name}\t{docnum}"
    )
    
    # SPL lines (one per item)
    for r in rows:
        item_code = get_qb_item_code(r)
        
        # Parse numeric values (handle formatted strings like "1,234.56")
        try:
            qty = float(str(r.get("quantity", 0)).replace(",", ""))
        except (ValueError, TypeError):
            qty = 0
        
        try:
            price = float(str(r.get("pi_unit_price", 0)).replace(",", ""))
        except (ValueError, TypeError):
            price = 0
        
        try:
            amount = float(str(r.get("pi_amount", 0)).replace(",", ""))
        except (ValueError, TypeError):
            amount = qty * price
        
        lines.append(
            f"SPL\tSALESORD\t{txn_date}\tSales\t{customer_name}\t{qty}\t{price}\t{amount}\t{item_code}"
        )
    
    lines.append("ENDTRNS")
    
    return "\n".join(lines)
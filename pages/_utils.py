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


def get_qb_item_code(row, qb_items=[]):
    """
    Get QuickBooks item code from row.
    Prioritizes type_code (from metadata), falls back to techpia_code.
    """
    temp = ''
    if "type_code" in row:
        temp = row["type_code"]
    if "techpia_code" in row:
        temp = row["techpia_code"]
    if "description" in row:
        temp = row["description"]
    if temp != '' and qb_items != []:
        temp_ = [x for x in qb_items if temp in x]
        if temp_:
            return temp_[0]
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

def date_Format_convert(x):
    from datetime import datetime

    y, m, d = x.split("/")
    return f"{m}/{d}/{y}"
    
def validate_items_against_qb(rows, qb_items):
    """
    Validate that all item codes in rows exist in QuickBooks.
    Returns list of missing items.
    """
    missing = []
    for r in rows:
        code = get_qb_item_code(r, qb_items)
        if code and code not in qb_items:
            missing.append(f"{code} ({r.get('description', 'No description')})")
    return missing

def generate_purchase_order_iif(rows, qb_items, vendor_name, container=False, txn_date=None, docnum=50001):
    """
    Generate IIF content for a Purchase Order.

    Fix:
    - Use INVITEM instead of ITEM
    - INVITEM must appear before QNTY
    """

    if txn_date is None:
        txn_date = qb_date()

    lines = [
        "!TRNS\tTRNSTYPE\tDATE\tACCNT\tNAME\tDOCNUM",
        "!SPL\tTRNSTYPE\tDATE\tACCNT\tNAME\tINVITEM\tQNTY\tPRICE\tAMOUNT",
        "!ENDTRNS",
    ]

    # TRNS header line
    lines.append(
        f"TRNS\tPURCHORD\t{txn_date}\tAccounts Payable\t{vendor_name}\t{docnum}"
    )

    unit_price, amount = ["price", "po_unit_price"][container==True], ["amount", "po_amount"][container==True]
    for r in rows:
        
        item_code = get_qb_item_code(r, qb_items)
        try:
            qty = float(str(r.get("quantity", 0)).replace(",", ""))
        except (ValueError, TypeError):
            qty = 0

        try:
            price = float(str(r[unit_price]).replace(",", ""))
        except (ValueError, TypeError):
            price = 0

        try:
            amount = float(str(r[amount]).replace(",", ""))
        except (ValueError, TypeError):
            amount = qty * price

        try:
            date = date_Format_convert(str(r.get("date", 0)).replace(",", ""))
        except:
            date = txn_date

        lines.append(
            f"SPL\tPURCHORD\t{date}\tInventory Asset\t{vendor_name}\t{item_code}\t{qty}\t{price}\t{amount}"
        )

    lines.append("ENDTRNS")

    return "\n".join(lines)


def generate_sales_order_iif(rows, qb_items, customer_name, container=False, txn_date=None, docnum=10001):

    if txn_date is None:
        txn_date = qb_date()

    lines = [
        "!TRNS\tTRNSTYPE\tDATE\tACCNT\tNAME\tDOCNUM",
        "!SPL\tTRNSTYPE\tDATE\tACCNT\tNAME\tQNTY\tPRICE\tAMOUNT\tITEM",
        "!ENDTRNS",
    ]

    lines.append(
        f"TRNS\tSALESORDER\t{txn_date}\tAccounts Receivable\t{customer_name}\t{docnum}"
    )
    unit_price, amount = ["price", "pi_unit_price"][container==True], ["amount", "pi_amount"][container==True]
    for r in rows:

        item_code = get_qb_item_code(r, qb_items)

        try:
            qty = float(str(r.get("quantity", 0)).replace(",", ""))
        except:
            qty = 0

        try:
            price = float(str(r[unit_price]).replace(",", ""))
        except:
            price = 0

        try:
            amount = float(str(r[amount]).replace(",", ""))
        except:
            amount = qty * price

        try:
            date = date_Format_convert(str(r.get("date", 0)).replace(",", ""))
        except:
            date = txn_date

        lines.append(
            f"SPL\tSALESORDER\t{date}\tSales\t{customer_name}\t{qty}\t{price}\t{amount}\t{item_code}"
        )

    lines.append("ENDTRNS")

    return "\n".join(lines)

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


def qb_item_name(row):
    base = f"{row['item']} {row['vlt']}%"
    if row.get("composition"):
        comp = "/".join(map(str, row["composition"]))
        return f"{base} {row['width']}\" ({comp})"
    return f"{base} {row['width']}\""

def validate_against_qb(final_rows, items, vendors, customers):
    missing_items = set()
    missing_vendors = set()
    missing_customers = set()

    for r in final_rows:
        item = qb_item_name(r)
        if item not in items:
            missing_items.add(item)

        if r["vendor"] not in vendors:
            missing_vendors.add(r["vendor"])

        if r["customer"] not in customers:
            missing_customers.add(r["customer"])

    if missing_items or missing_vendors or missing_customers:
        raise ValueError(
            f"QuickBooks lookup failed:\n"
            f"Missing Items: {sorted(missing_items)}\n"
            f"Missing Vendors: {sorted(missing_vendors)}\n"
            f"Missing Customers: {sorted(missing_customers)}"
        )


from datetime import datetime

def qb_date(d):
    if isinstance(d, str):
        return d
    return d.strftime("%m/%d/%Y")

def item_name(row):
    """
    Build a stable QuickBooks item name
    """
    base = f"{row['item']} {row['vlt']}%"
    if row.get("composition"):
        comp = "/".join(map(str, row["composition"]))
        return f"{base} {row['width']}\" ({comp})"
    return f"{base} {row['width']}\""

def generate_sales_order_iif(rows, outfile):
    lines = [
        "!TRNS\tTRNSTYPE\tDATE\tACCNT\tNAME\tDOCNUM",
        "!SPL\tTRNSTYPE\tDATE\tACCNT\tNAME\tQNTY\tPRICE\tAMOUNT\tITEM",
        "!ENDTRNS",
    ]

    docnum = 10001

    for r in rows:
        item = qb_item_name(r)
        amount = r["qty"] * r["price"]

        lines.append(
            f"TRNS\tSALESORD\t{r['txn_date']}\tAccounts Receivable\t"
            f"{r['customer']}\t{docnum}"
        )

        lines.append(
            f"SPL\tSALESORD\t{r['txn_date']}\tSales\t"
            f"{r['customer']}\t{r['qty']}\t{r['price']}\t{amount}\t{item}"
        )

        lines.append("ENDTRNS")
        docnum += 1

    with open(outfile, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def generate_purchase_order_iif(rows, outfile):
    lines = [
        "!TRNS\tTRNSTYPE\tDATE\tACCNT\tNAME\tDOCNUM",
        "!SPL\tTRNSTYPE\tDATE\tACCNT\tNAME\tQNTY\tPRICE\tAMOUNT\tITEM",
        "!ENDTRNS",
    ]

    docnum = 50001

    for r in rows:
        item = qb_item_name(r)
        price = r.get("vendor_price", r["price"])
        amount = r["qty"] * price * -1

        lines.append(
            f"TRNS\tPO\t{r['txn_date']}\tAccounts Payable\t"
            f"{r['vendor']}\t{docnum}"
        )

        lines.append(
            f"SPL\tPO\t{r['txn_date']}\tCost of Goods Sold\t"
            f"{r['vendor']}\t{r['qty']}\t{price}\t{amount}\t{item}"
        )

        lines.append("ENDTRNS")
        docnum += 1

    with open(outfile, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    QB_IIF_PATH = "smp.iif"

    existing_items, existing_vendors, existing_customers = load_qb_lists_from_iif(QB_IIF_PATH)

    validate_against_qb(
        final_rows,
        existing_items,
        existing_vendors,
        existing_customers
    )

    generate_sales_order_iif(final_rows, "/mnt/data/sales_orders.iif")
    generate_purchase_order_iif(final_rows, "/mnt/data/purchase_orders.iif")
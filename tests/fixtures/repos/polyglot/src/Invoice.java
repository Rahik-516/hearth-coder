package billing;

import java.math.BigDecimal;
import java.util.List;

/** A customer invoice. */
public class Invoice {
    private final String id;
    private final List<LineItem> lines;

    public Invoice(String id, List<LineItem> lines) {
        this.id = id;
        this.lines = lines;
    }

    public BigDecimal computeSubtotal() {
        BigDecimal total = BigDecimal.ZERO;
        for (LineItem line : lines) {
            total = total.add(line.amount());
        }
        return total;
    }

    public String getId() {
        return id;
    }
}

interface LineItem {
    BigDecimal amount();
}

import type { CartItem, CartTotals, Currency, Money } from "./types";
import { CartStatus } from "./types";
import { RateLimiter } from "./rateLimiter";

export class CartError extends Error {
  constructor(
    message: string,
    readonly code: string,
  ) {
    super(message);
    this.name = "CartError";
  }
}

const ZERO: Money = { amount: 0, currency: "USD" };

/** Coordinates cart contents and totalling. */
export class ShoppingCart {
  private items: CartItem[] = [];
  private status: CartStatus = CartStatus.Open;

  constructor(
    private readonly currency: Currency = "USD",
    private readonly limiter: RateLimiter = new RateLimiter(20, 5),
  ) {}

  add(item: CartItem): void {
    if (this.status !== CartStatus.Open) {
      throw new CartError("cart is not open", "CART_LOCKED");
    }
    if (item.quantity <= 0) {
      throw new CartError("quantity must be positive", "BAD_QUANTITY");
    }
    this.limiter.acquire();
    this.items.push(item);
  }

  remove(sku: string): boolean {
    const before = this.items.length;
    this.items = this.items.filter((entry) => entry.sku !== sku);
    return this.items.length !== before;
  }

  /** Total the cart, applying a percentage discount. */
  finalize(discountPercent = 0): CartTotals {
    if (this.items.length === 0) {
      throw new CartError("cannot finalize an empty cart", "EMPTY_CART");
    }

    const subtotal = this.items.reduce(
      (sum, item) => sum + item.unitPrice.amount * item.quantity,
      0,
    );
    const discount = round(subtotal * (discountPercent / 100));

    this.status = CartStatus.Locked;

    return {
      subtotal: { amount: round(subtotal), currency: this.currency },
      discount: { amount: discount, currency: this.currency },
      total: { amount: round(subtotal - discount), currency: this.currency },
    };
  }

  get size(): number {
    return this.items.length;
  }
}

export function round(value: number): number {
  return Math.round(value * 100) / 100;
}

export const emptyTotals = (): CartTotals => ({
  subtotal: ZERO,
  discount: ZERO,
  total: ZERO,
});

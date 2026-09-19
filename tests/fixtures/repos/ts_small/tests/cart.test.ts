import { describe, expect, it } from "vitest";
import { ShoppingCart, round } from "../src/cart";
import type { CartItem } from "../src/types";

const widget: CartItem = {
  sku: "W-1",
  title: "Widget",
  quantity: 2,
  unitPrice: { amount: 10, currency: "USD" },
};

describe("ShoppingCart", () => {
  it("totals its items", () => {
    const cart = new ShoppingCart();
    cart.add(widget);
    expect(cart.finalize().total.amount).toBe(20);
  });

  it("applies a discount", () => {
    const cart = new ShoppingCart();
    cart.add(widget);
    expect(cart.finalize(10).discount.amount).toBe(2);
  });

  it("refuses an empty cart", () => {
    expect(() => new ShoppingCart().finalize()).toThrow(/empty cart/);
  });
});

describe("round", () => {
  it("rounds to two places", () => {
    expect(round(1.005)).toBe(1.0);
  });
});

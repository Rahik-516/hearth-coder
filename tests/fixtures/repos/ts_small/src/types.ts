/** Domain types for the cart fixture. */

export type Currency = "USD" | "EUR" | "GBP";

export interface Money {
  amount: number;
  currency: Currency;
}

export interface CartItem {
  sku: string;
  title: string;
  quantity: number;
  unitPrice: Money;
}

export enum CartStatus {
  Open = "open",
  Locked = "locked",
  Abandoned = "abandoned",
}

export type CartTotals = {
  subtotal: Money;
  discount: Money;
  total: Money;
};

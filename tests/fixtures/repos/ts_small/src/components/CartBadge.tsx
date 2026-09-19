import { ShoppingCart } from "../cart";

interface CartBadgeProps {
  cart: ShoppingCart;
  label?: string;
}

export function CartBadge({ cart, label = "Items" }: CartBadgeProps) {
  const count = cart.size;
  return (
    <span className="cart-badge" data-count={count}>
      {label}: {count}
    </span>
  );
}

export default CartBadge;

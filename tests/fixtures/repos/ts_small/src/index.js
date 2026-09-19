// Plain JavaScript, so the fixture exercises the javascript grammar too.
const { ShoppingCart } = require("./cart");

function createCart(currency) {
  return new ShoppingCart(currency);
}

const DEFAULT_CURRENCY = "USD";

module.exports = { createCart, DEFAULT_CURRENCY };

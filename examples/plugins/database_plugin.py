"""
DatabasePlugin - Demonstrates database-like operations.

This plugin simulates database queries to show how AI can interact with data stores.
In a real application, these would connect to actual databases (PostgreSQL, MongoDB, etc.).
"""

from semantic_kernel.functions import kernel_function
from typing import Annotated


class DatabasePlugin:
    """A plugin that provides database operations (simulated)."""
    
    # Simulated database tables
    _users = [
        {"id": 1, "name": "Alice Johnson", "email": "alice@example.com", "role": "Admin", "active": True},
        {"id": 2, "name": "Bob Smith", "email": "bob@example.com", "role": "User", "active": True},
        {"id": 3, "name": "Charlie Brown", "email": "charlie@example.com", "role": "User", "active": False},
        {"id": 4, "name": "Diana Prince", "email": "diana@example.com", "role": "Manager", "active": True},
        {"id": 5, "name": "Eve Davis", "email": "eve@example.com", "role": "User", "active": True},
    ]
    
    _products = [
        {"id": 101, "name": "Laptop", "price": 999.99, "stock": 15, "category": "Electronics"},
        {"id": 102, "name": "Mouse", "price": 29.99, "stock": 50, "category": "Electronics"},
        {"id": 103, "name": "Keyboard", "price": 79.99, "stock": 30, "category": "Electronics"},
        {"id": 104, "name": "Monitor", "price": 299.99, "stock": 8, "category": "Electronics"},
        {"id": 105, "name": "Desk Chair", "price": 199.99, "stock": 12, "category": "Furniture"},
        {"id": 106, "name": "Desk", "price": 399.99, "stock": 5, "category": "Furniture"},
    ]
    
    _orders = [
        {"id": 1001, "user_id": 1, "product_id": 101, "quantity": 1, "status": "Delivered"},
        {"id": 1002, "user_id": 2, "product_id": 102, "quantity": 2, "status": "Shipped"},
        {"id": 1003, "user_id": 1, "product_id": 104, "quantity": 1, "status": "Processing"},
        {"id": 1004, "user_id": 4, "product_id": 105, "quantity": 3, "status": "Delivered"},
    ]
    
    @kernel_function(
        name="get_user_by_id",
        description="Retrieves user information by user ID."
    )
    def get_user_by_id(
        self,
        user_id: Annotated[int, "The user ID to look up"],
    ) -> Annotated[str, "User information"]:
        """Get user by ID."""
        print(f"[DatabasePlugin.get_user_by_id] Looking up user {user_id}")
        
        user = next((u for u in self._users if u["id"] == user_id), None)
        if user:
            return (
                f"User ID: {user['id']}\n"
                f"Name: {user['name']}\n"
                f"Email: {user['email']}\n"
                f"Role: {user['role']}\n"
                f"Active: {user['active']}"
            )
        return f"User with ID {user_id} not found."
    
    @kernel_function(
        name="get_user_by_name",
        description="Searches for a user by name (partial match supported)."
    )
    def get_user_by_name(
        self,
        name: Annotated[str, "The name or partial name to search for"],
    ) -> Annotated[str, "User information"]:
        """Search user by name."""
        print(f"[DatabasePlugin.get_user_by_name] Searching for user: {name}")
        
        name_lower = name.lower()
        matching_users = [u for u in self._users if name_lower in u["name"].lower()]
        
        if not matching_users:
            return f"No users found matching '{name}'."
        
        result = f"Found {len(matching_users)} user(s):\n\n"
        for user in matching_users:
            result += (
                f"ID: {user['id']}, Name: {user['name']}, "
                f"Email: {user['email']}, Role: {user['role']}\n"
            )
        return result
    
    @kernel_function(
        name="get_product_by_id",
        description="Retrieves product information by product ID."
    )
    def get_product_by_id(
        self,
        product_id: Annotated[int, "The product ID to look up"],
    ) -> Annotated[str, "Product information"]:
        """Get product by ID."""
        print(f"[DatabasePlugin.get_product_by_id] Looking up product {product_id}")
        
        product = next((p for p in self._products if p["id"] == product_id), None)
        if product:
            return (
                f"Product ID: {product['id']}\n"
                f"Name: {product['name']}\n"
                f"Price: ${product['price']}\n"
                f"Stock: {product['stock']} units\n"
                f"Category: {product['category']}"
            )
        return f"Product with ID {product_id} not found."
    
    @kernel_function(
        name="search_products",
        description="Searches for products by name or category."
    )
    def search_products(
        self,
        query: Annotated[str, "The search term (name or category)"],
    ) -> Annotated[str, "List of matching products"]:
        """Search products."""
        print(f"[DatabasePlugin.search_products] Searching for: {query}")

        query_lower = query.lower()
        matching_products = [
            p for p in self._products
            if query_lower in p["name"].lower() or query_lower in p["category"].lower()
        ]

        if not matching_products:
            return f"No products found matching '{query}'."

        result = f"Found {len(matching_products)} product(s):\n\n"
        for product in matching_products:
            result += (
                f"ID: {product['id']}, Name: {product['name']}, "
                f"Price: ${product['price']}, Stock: {product['stock']}\n"
            )
        return result

    @kernel_function(
        name="get_user_orders",
        description="Gets all orders for a specific user by user ID."
    )
    def get_user_orders(
        self,
        user_id: Annotated[int, "The user ID to get orders for"],
    ) -> Annotated[str, "List of user's orders"]:
        """Get orders for a user."""
        print(f"[DatabasePlugin.get_user_orders] Getting orders for user {user_id}")

        user_orders = [o for o in self._orders if o["user_id"] == user_id]

        if not user_orders:
            return f"No orders found for user ID {user_id}."

        result = f"Orders for User ID {user_id}:\n\n"
        for order in user_orders:
            product = next((p for p in self._products if p["id"] == order["product_id"]), None)
            product_name = product["name"] if product else "Unknown"
            result += (
                f"Order ID: {order['id']}, Product: {product_name}, "
                f"Quantity: {order['quantity']}, Status: {order['status']}\n"
            )
        return result

    @kernel_function(
        name="check_stock",
        description="Checks if a product has sufficient stock available."
    )
    def check_stock(
        self,
        product_id: Annotated[int, "The product ID to check"],
        quantity: Annotated[int, "The quantity needed"],
    ) -> Annotated[str, "Stock availability information"]:
        """Check product stock."""
        print(f"[DatabasePlugin.check_stock] Checking stock for product {product_id}")

        product = next((p for p in self._products if p["id"] == product_id), None)
        if not product:
            return f"Product with ID {product_id} not found."

        if product["stock"] >= quantity:
            return f"✓ {product['name']} has {product['stock']} units in stock. {quantity} units available."
        else:
            return f"✗ {product['name']} only has {product['stock']} units in stock. Cannot fulfill order for {quantity} units."

    @kernel_function(
        name="get_total_inventory_value",
        description="Calculates the total value of all products in inventory."
    )
    def get_total_inventory_value(self) -> Annotated[str, "Total inventory value"]:
        """Calculate total inventory value."""
        print(f"[DatabasePlugin.get_total_inventory_value] Calculating inventory value")

        total_value = sum(p["price"] * p["stock"] for p in self._products)

        result = "Inventory Summary:\n\n"
        for product in self._products:
            value = product["price"] * product["stock"]
            result += f"{product['name']}: {product['stock']} units × ${product['price']} = ${value:.2f}\n"
        result += f"\nTotal Inventory Value: ${total_value:.2f}"

        return result


# Project Kickoff Notes: Stage 1 Architecture

## 1. Division of Labor
We are splitting Stage 1 into two main domains to keep our code modular:
*   **World & Planner** (State management, physics, and execution)
*   **Parser** (NLTK grammar, natural language understanding, and intent resolution)

## 2. The Main Loop (Traffic Cop)
To keep the Planner and Parser completely decoupled, a **Main Loop** will act as the controller:
1.  Reads user input and calls the **Parser**.
2.  Checks the `status` of the Parser's output:
    *   If **`RESOLVED`**: Passes the parsed arguments to the **Planner** to execute the action, then waits for the next command.
    *   If **`AMBIGUOUS`** (multiple matches) or **`UNRECOGNIZED`** (parsing error): Skips the Planner and immediately reprompts the user for clarification.

*(Note: The exact JSON API contract for communication from the Parser to the Planner is defined in the `api.md` file.)*

## 3. Assignment Requirements (Minimums Met)
To satisfy the "Minimum World Complexity" for this assignment, we need:
*   **5 Actions:** `PICKUP`, `PLACE`, `OPEN`, `CLOSE`, `INSPECT`
*   **4 Attribute Types:** `SHAPE` (must include boxes), `COLOR`, `SIZE`, `MATERIAL`, `STATE` (open or closed, applies to boxes)
*   **2 Locations/Zones:** `TABLE`, `FLOOR`
*   **12 Objects:** Any 12 objects with varying attributes.

## 4. The Query API (Second Contract)
To solve the symbol grounding problem, `world.py` must implement a query function that the Parser can call to find objects. It must use this exact format:
```python
def find_objects(
    self, 
    shape: Optional[str] = None, 
    color: Optional[str] = None, 
    size: Optional[str] = None, 
    material: Optional[str] = None, 
    state: Optional[str] = None, # open or closed
    location_id: Optional[str] = None # One of the two locations (table or floor)
) -> list[str]: # returns a list of matching object ids
```
*   **Rule 1:** If no arguments are passed, it returns all objects.
*   **Rule 2:** If multiple arguments are passed, it acts as an `AND` filter (e.g., must be BOTH red AND a box).

## Next Steps
*   **World Builder:** Look at the original shrdlu.ipynb notebook for the baseline Obj and World classes, but expand them to include the new attributes and the find_objects query function.
*   **Linguist:** Begin writing the NLTK grammar to map natural language to the structured JSON payload (found in `api.md`).

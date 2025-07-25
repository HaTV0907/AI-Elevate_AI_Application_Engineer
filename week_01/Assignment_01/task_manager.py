# Step 1: Define the Data Structure
tasks = []
 
# Step 2: Implement Core Functions
 
def add_task(description):
    """Adds a new task to the tasks list."""
    # Generate a unique task ID. If tasks list is empty, start with 1, otherwise increment the last task's ID.
    task_id = tasks[-1]["id"] + 1 if tasks else 1
    task = {"id": task_id, "description": description, "completed": False}
    tasks.append(task)
    print(f"Task '{description}' added with ID {task_id}.")
 
def view_tasks():
    """Displays all tasks with their ID, description, and completion status."""
    if not tasks:
        print("No tasks available.")
        return
 
    print("\n--- Your Tasks ---")
    for task in tasks:
        status = "Done" if task["completed"] else "Pending"
        print(f"{task['id']}: {task['description']} [{status}]")
    print("------------------")
 
def mark_completed(task_id):
    """Marks a task as completed given its ID."""
    found = False
    for task in tasks:
        if task["id"] == task_id:
            task["completed"] = True
            print(f"Task ID {task_id} marked as completed.")
            found = True
            break
    if not found:
        print(f"No task found with ID {task_id}.")
 
def delete_task(task_id):
    """Deletes a task given its ID."""
    global tasks # Declare 'tasks' as global to modify the list directly
    initial_task_count = len(tasks)
    tasks = [task for task in tasks if task["id"] != task_id]
    if len(tasks) < initial_task_count:
        print(f"Task ID {task_id} deleted.")
    else:
        print(f"No task found with ID {task_id}.")
 
# Step 3: Create the User Interface Loop
 
def main():
    """Main function to run the command-line task manager."""
    print("Welcome to the Command-Line Task Manager!")
    while True:
        print("\nOptions:")
        print("  add      - Add a new task")
        print("  view     - View all tasks")
        print("  complete - Mark a task as completed")
        print("  delete   - Delete a task")
        print("  exit     - Exit the application")
       
        choice = input("Enter your command: ").strip().lower()
 
        if choice == "add":
            desc = input("Enter task description: ").strip()
            if desc: # Ensure description is not empty
                add_task(desc)
            else:
                print("Task description cannot be empty.")
        elif choice == "view":
            view_tasks()
        elif choice == "complete":
            try:
                task_id = int(input("Enter task ID to mark complete: "))
                mark_completed(task_id)
            except ValueError:
                print("Invalid input. Please enter a numerical task ID.")
        elif choice == "delete":
            try:
                task_id = int(input("Enter task ID to delete: "))
                delete_task(task_id)
            except ValueError:
                print("Invalid input. Please enter a numerical task ID.")
        elif choice == "exit":
            print("Exiting Task Manager. Goodbye!")
            break
        else:
            print("Invalid command. Please choose from 'add', 'view', 'complete', 'delete', or 'exit'.")
 
# Step 4: Run and Test
if __name__ == "__main__":
    main()
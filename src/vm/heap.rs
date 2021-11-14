use std::cell::RefCell;
use std::ops::{Deref, DerefMut};
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub enum Object {
    String(String),
}

// Internal representation of a heap object. The objects are arranged in a linked list using the `next` field, and the objects
// are owned by the nodes (with the head node owned by the Heap itself)
#[derive(Debug)]
struct HeapNode {
    next: Option<Box<HeapNode>>,
    object: Rc<RefCell<Object>>,
}

// We hide the implementation details of the shared ownership + interior mutability behind this newtype.
// However, the `borrow` and `borrow_mut` methods mirror (and delegate to) the RefCell methods of the same name.
#[derive(Debug)]
pub struct SharedObject(Rc<RefCell<Object>>);

impl SharedObject {
    pub fn borrow(&self) -> impl Deref<Target = Object> + '_ {
        self.0.borrow()
    }

    pub fn borrow_mut(&self) -> impl DerefMut<Target = Object> + '_ {
        self.0.borrow_mut()
    }
}

#[derive(Debug, Clone)]
pub struct HeapRef {
    value: Weak<RefCell<Object>>,
}

#[derive(Debug)]
/// A type for allocating, tracking, and eventually GCing heap-allocated Lox object
pub struct Heap {
    head: Option<Box<HeapNode>>,
}

impl Heap {
    /// A new, empty heap.
    pub fn new() -> Heap {
        Heap { head: None }
    }

    /// Add an empty string to the heap, and return the node by which it can be accessed.
    /// If you want to give the string a value ,us `new_string_with_value`, but this method
    /// doesn't do the String's heap allocation (as it uses `String::new()`).
    // I tried for a while to return a value that has a lifetime tied to lifetime of the heap,
    // but couldn't make it work :(.
    pub fn new_string(&mut self) -> HeapRef {
        let obj_in_rc = Rc::new(RefCell::new(Object::String(String::new())));
        let obj = HeapNode {
            object: obj_in_rc.clone(),
            next: self.head.take(),
        };
        self.head = Some(Box::new(obj));
        HeapRef {
            value: std::rc::Rc::downgrade(&obj_in_rc),
        }
    }

    /// Add an non-empty string to the heap, copying from value, and return the node by which it can be accessed.
    pub fn new_string_with_value(&mut self, value: &str) -> HeapRef {
        let obj_in_rc = Rc::new(RefCell::new(Object::String(String::from(value))));
        let obj = HeapNode {
            object: obj_in_rc.clone(),
            next: self.head.take(),
        };
        self.head = Some(Box::new(obj));
        HeapRef {
            value: std::rc::Rc::downgrade(&obj_in_rc),
        }
    }

    // Print out all the objects on the heap in linked list order for debugging.
    #[allow(dead_code)]
    fn dump(&self) {
        let mut next = &self.head;
        while let Some(node) = next {
            match &*node.object.borrow() {
                Object::String(s) => {
                    println!("{}", s);
                }
            }
            next = &node.next
        }
    }
}

impl Default for Heap {
    fn default() -> Self {
        Self::new()
    }
}

impl Object {
    pub fn is_string(&self) -> bool {
        true
    }

    pub fn as_string(&self) -> Option<&String> {
        let Object::String(s) = self;
        Some(s)
    }

    pub fn as_string_mut(&mut self) -> Option<&mut String> {
        let Object::String(s) = self;
        Some(s)
    }
}

impl HeapRef {
    // Convert a ref to an object, panicking if the ref is no longer alive.
    pub fn as_obj(&self) -> SharedObject {
        SharedObject(self.value.upgrade().unwrap())
    }

    // Apply a function to the inner object if it's a string, returning the result if it's a string
    // And none if it isn't a string
    pub fn map_as_string<F, Ret>(&self, f: F) -> Option<Ret>
    where
        F: FnOnce(&String) -> Ret,
    {
        Some(f(self.as_obj().borrow().as_string()?))
    }

    // Same as map_as_string but with mutability.
    pub fn map_as_string_mut<F, Ret>(&self, f: F) -> Option<Ret>
    where
        F: FnOnce(&mut String) -> Ret,
    {
        Some(f(self.as_obj().borrow_mut().as_string_mut()?))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_heap() {
        let mut heap = Heap::new();
        let node = heap.new_string();
        node.as_obj()
            .borrow_mut()
            .as_string_mut()
            .unwrap()
            .push_str("Goodbye, world");

        heap.new_string_with_value("Hello, world");

        heap.dump();
    }
}

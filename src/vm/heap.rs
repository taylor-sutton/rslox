use std::cell::RefCell;
use std::collections::HashSet;
use std::ops::{Deref, DerefMut};
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub enum Object {
    InternedString { length: usize, data: *const u8 },
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
    string_table: HashSet<String>,
}

impl Heap {
    /// A new, empty heap.
    pub fn new() -> Heap {
        Heap {
            head: None,
            string_table: HashSet::new(),
        }
    }

    /// New string with value, taking ownership and interning it.
    pub fn new_string(&mut self, value: String) -> HeapRef {
        // Ideally we have the unstable method HashSet::get_or_insert, but alas
        self.string_table.insert(value.clone());
        let string_ref = self.string_table.get(&value).unwrap();
        let o = Rc::new(RefCell::new(Object::InternedString {
            data: string_ref.as_ptr(),
            length: string_ref.len(),
        }));
        let obj = HeapNode {
            object: o.clone(),
            next: self.head.take(),
        };
        self.head = Some(Box::new(obj));
        HeapRef {
            value: std::rc::Rc::downgrade(&o),
        }
    }

    // Print out all the objects on the heap in linked list order for debugging.
    #[allow(dead_code)]
    fn dump(&self) {
        let mut next = &self.head;
        while let Some(node) = next {
            match &*node.object.borrow() {
                Object::InternedString { length, data } => {
                    let s: &str = unsafe {
                        let data: &[u8] = std::slice::from_raw_parts(*data, *length);
                        std::str::from_utf8_unchecked(data)
                    };
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

    fn as_string(&self) -> Option<&str> {
        match self {
            Object::InternedString { data, length } => {
                let s: &str = unsafe {
                    let data: &[u8] = std::slice::from_raw_parts(*data, *length);
                    std::str::from_utf8_unchecked(data)
                };
                Some(s)
            }
        }
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
        F: FnOnce(&str) -> Ret,
    {
        Some(f(self.as_obj().borrow().as_string()?))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_heap() {
        let mut heap = Heap::new();
        let n1 = heap.new_string("Goodbye, world".into());
        let n2 = heap.new_string("Hello, world".into());

        n1.map_as_string(|s| assert_eq!(s, "Goodbye, world"));
        n2.map_as_string(|s| assert_eq!(s, "Hello, world"));

        heap.dump();
    }
}

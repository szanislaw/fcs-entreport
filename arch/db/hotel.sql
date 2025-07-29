-- ========================
-- 1. Create Tables
-- ========================

CREATE TABLE regions (
    region_id INTEGER PRIMARY KEY,
    region_name TEXT NOT NULL
);

CREATE TABLE hotels (
    hotel_id INTEGER PRIMARY KEY,
    region_id INTEGER NOT NULL,
    hotel_name TEXT NOT NULL,
    city TEXT NOT NULL,
    rating REAL,
    FOREIGN KEY (region_id) REFERENCES regions(region_id)
);

CREATE TABLE rooms (
    room_id INTEGER PRIMARY KEY,
    hotel_id INTEGER NOT NULL,
    room_type TEXT NOT NULL,
    base_price REAL NOT NULL,
    is_available BOOLEAN DEFAULT 1,
    FOREIGN KEY (hotel_id) REFERENCES hotels(hotel_id)
);

CREATE TABLE guests (
    guest_id INTEGER PRIMARY KEY,
    guest_name TEXT NOT NULL,
    nationality TEXT NOT NULL,
    email TEXT,
    phone TEXT
);

CREATE TABLE bookings (
    booking_id INTEGER PRIMARY KEY,
    room_id INTEGER NOT NULL,
    guest_id INTEGER NOT NULL,
    check_in DATE NOT NULL,
    check_out DATE NOT NULL,
    total_price REAL NOT NULL,
    booking_status TEXT CHECK (booking_status IN ('Confirmed', 'Cancelled', 'Completed')) DEFAULT 'Confirmed',
    FOREIGN KEY (room_id) REFERENCES rooms(room_id),
    FOREIGN KEY (guest_id) REFERENCES guests(guest_id)
);

CREATE TABLE payments (
    payment_id INTEGER PRIMARY KEY,
    booking_id INTEGER NOT NULL,
    amount REAL NOT NULL,
    payment_date DATE NOT NULL,
    payment_method TEXT CHECK (payment_method IN ('Credit Card', 'Cash', 'Online')),
    FOREIGN KEY (booking_id) REFERENCES bookings(booking_id)
);

CREATE TABLE staff (
    staff_id INTEGER PRIMARY KEY,
    staff_name TEXT NOT NULL,
    position TEXT NOT NULL,
    hire_date DATE NOT NULL
);

CREATE TABLE hotel_staff (
    hotel_id INTEGER NOT NULL,
    staff_id INTEGER NOT NULL,
    PRIMARY KEY (hotel_id, staff_id),
    FOREIGN KEY (hotel_id) REFERENCES hotels(hotel_id),
    FOREIGN KEY (staff_id) REFERENCES staff(staff_id)
);

CREATE TABLE services (
    service_id INTEGER PRIMARY KEY,
    service_name TEXT NOT NULL,
    cost REAL
);

CREATE TABLE room_services (
    room_id INTEGER NOT NULL,
    service_id INTEGER NOT NULL,
    PRIMARY KEY (room_id, service_id),
    FOREIGN KEY (room_id) REFERENCES rooms(room_id),
    FOREIGN KEY (service_id) REFERENCES services(service_id)
);

CREATE TABLE reviews (
    review_id INTEGER PRIMARY KEY,
    guest_id INTEGER NOT NULL,
    hotel_id INTEGER NOT NULL,
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    comment TEXT,
    review_date DATE NOT NULL,
    FOREIGN KEY (guest_id) REFERENCES guests(guest_id),
    FOREIGN KEY (hotel_id) REFERENCES hotels(hotel_id)
);

-- ========================
-- 2. Insert Data
-- ========================

INSERT INTO regions VALUES
(1, 'Asia'), (2, 'Europe'), (3, 'America');

INSERT INTO hotels VALUES
(101, 1, 'Marina Bay Sands', 'Singapore', 4.7),
(102, 1, 'Hotel Tokyo Deluxe', 'Tokyo', 4.3),
(201, 2, 'Grand Berlin Palace', 'Berlin', 4.5),
(202, 2, 'Paris Stay Inn', 'Paris', 4.1),
(301, 3, 'New York Tower Inn', 'New York', 4.6),
(302, 3, 'Sunset LA Resort', 'Los Angeles', 4.0);

INSERT INTO rooms VALUES
(1, 101, 'Deluxe', 200.00, 1),
(2, 101, 'Standard', 150.00, 1),
(3, 102, 'Deluxe', 210.00, 0),
(4, 201, 'Standard', 140.00, 1),
(5, 301, 'Suite', 300.00, 1);

INSERT INTO guests VALUES
(5001, 'Alice Tan', 'Singaporean', 'alice@example.com', '+65-12345678'),
(5002, 'John Muller', 'German', 'john@example.com', '+49-98765432'),
(5003, 'Pierre Dubois', 'French', 'pierre@example.com', '+33-76543210'),
(5004, 'Mike Chen', 'American', 'mike@example.com', '+1-202-555-0123');

INSERT INTO bookings VALUES
(1001, 1, 5001, '2025-07-19', '2025-07-21', 400.00, 'Completed'),
(1002, 2, 5002, '2025-07-20', '2025-07-21', 150.00, 'Completed'),
(1003, 4, 5003, '2025-07-20', '2025-07-22', 280.00, 'Cancelled'),
(1004, 5, 5004, '2025-07-20', '2025-07-22', 600.00, 'Confirmed');

INSERT INTO payments VALUES
(2001, 1001, 400.00, '2025-07-18', 'Credit Card'),
(2002, 1002, 150.00, '2025-07-19', 'Cash'),
(2003, 1004, 600.00, '2025-07-20', 'Online');

INSERT INTO staff VALUES
(9001, 'Sarah Lim', 'Manager', '2020-01-01'),
(9002, 'Tom Becker', 'Chef', '2019-03-15'),
(9003, 'Linda Zhang', 'Receptionist', '2021-07-10');

INSERT INTO hotel_staff VALUES
(101, 9001),
(101, 9003),
(102, 9002);

INSERT INTO services VALUES
(1, 'Room Service', 20.00),
(2, 'Laundry', 10.00),
(3, 'Spa Access', 50.00);

INSERT INTO room_services VALUES
(1, 1),
(1, 2),
(2, 1),
(5, 3);

INSERT INTO reviews VALUES
(3001, 5001, 101, 5, 'Excellent stay!', '2025-07-22'),
(3002, 5002, 101, 4, 'Good service but a bit pricey.', '2025-07-23'),
(3003, 5003, 201, 2, 'Room was too small.', '2025-07-24');

-- ========================
-- 1. Create Tables
-- ========================
CREATE TABLE regions (
    region_id INT PRIMARY KEY,
    region_name VARCHAR(50)
);

CREATE TABLE hotels (
    hotel_id INT PRIMARY KEY,
    region_id INT,
    hotel_name VARCHAR(100),
    city VARCHAR(100),
    FOREIGN KEY (region_id) REFERENCES regions(region_id)
);

CREATE TABLE rooms (
    room_id INT PRIMARY KEY,
    hotel_id INT,
    room_type VARCHAR(50),
    base_price DECIMAL(10, 2),
    FOREIGN KEY (hotel_id) REFERENCES hotels(hotel_id)
);

CREATE TABLE guests (
    guest_id INT PRIMARY KEY,
    guest_name VARCHAR(100),
    nationality VARCHAR(100)
);

CREATE TABLE bookings (
    booking_id INT PRIMARY KEY,
    room_id INT,
    check_in DATE,
    check_out DATE,
    guest_id INT,
    total_price DECIMAL(10, 2),
    FOREIGN KEY (room_id) REFERENCES rooms(room_id),
    FOREIGN KEY (guest_id) REFERENCES guests(guest_id)
);

-- ========================
-- 2. Insert Data
-- ========================
INSERT INTO regions (region_id, region_name) VALUES
(1, 'Asia'),
(2, 'Europe'),
(3, 'America');

INSERT INTO hotels (hotel_id, region_id, hotel_name, city) VALUES
(101, 1, 'Marina Bay Sands', 'Singapore'),
(102, 1, 'Hotel Tokyo Deluxe', 'Tokyo'),
(201, 2, 'Grand Berlin Palace', 'Berlin'),
(202, 2, 'Paris Stay Inn', 'Paris'),
(301, 3, 'New York Tower Inn', 'New York'),
(302, 3, 'Sunset LA Resort', 'Los Angeles');

INSERT INTO rooms (room_id, hotel_id, room_type, base_price) VALUES
(1, 101, 'Deluxe', 200.00),
(2, 101, 'Standard', 150.00),
(3, 102, 'Deluxe', 210.00),
(4, 201, 'Standard', 140.00),
(5, 301, 'Suite', 300.00);

INSERT INTO guests (guest_id, guest_name, nationality) VALUES
(5001, 'Alice Tan', 'Singaporean'),
(5002, 'John Muller', 'German'),
(5003, 'Pierre Dubois', 'French'),
(5004, 'Mike Chen', 'American');

INSERT INTO bookings (booking_id, room_id, check_in, check_out, guest_id, total_price) VALUES
(1001, 1, '2025-07-19', '2025-07-21', 5001, 400.00),
(1002, 2, '2025-07-20', '2025-07-21', 5002, 150.00),
(1003, 4, '2025-07-20', '2025-07-22', 5003, 280.00),
(1004, 5, '2025-07-20', '2025-07-22', 5004, 600.00);

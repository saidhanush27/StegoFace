-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Feb 17, 2025 at 06:25 AM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `faceMDT`
--

-- --------------------------------------------------------

--
-- Table structure for table `admin`
--

CREATE TABLE `admin` (
  `uname` varchar(10) NOT NULL,
  `password` varchar(10) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `admin`
--

INSERT INTO `admin` (`uname`, `password`) VALUES
('admin', 'admin');

-- --------------------------------------------------------

--
-- Table structure for table `filetrans`
--

CREATE TABLE `filetrans` (
  `id` int(50) NOT NULL auto_increment,
  `uname` varchar(50) NOT NULL,
  `image` varchar(100) NOT NULL,
  `message` varchar(100) NOT NULL,
  `key1` varchar(100) NOT NULL,
  `key2` varchar(100) NOT NULL,
  `fimage` varchar(50) NOT NULL,
  PRIMARY KEY  (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=4 ;

--
-- Dumping data for table `filetrans`
--

INSERT INTO `filetrans` (`id`, `uname`, `image`, `message`, `key1`, `key2`, `fimage`) VALUES
(1, 'sundar', 'images.jpeg', 'hai', 'fbd7b67b', 'afc6327f', 'static/uploads/fbd7b67bface.jpg'),
(2, 'sundar', 'School-ID-Card-Template.png', 'sample', 'c2150690', 'e36df638', 'static/uploads/c2150690face.jpg'),
(3, 'sundar', 'images.jpeg', 'file details', '291c6c8f', '64e988d2', 'static/uploads/291c6c8fface.jpg');

-- --------------------------------------------------------

--
-- Table structure for table `user`
--

CREATE TABLE `user` (
  `id` int(50) NOT NULL auto_increment,
  `name` varchar(100) NOT NULL,
  `gender` varchar(10) NOT NULL,
  `address` varchar(100) NOT NULL,
  `email` varchar(100) NOT NULL,
  `pnumber` varchar(10) NOT NULL,
  `uname` varchar(10) NOT NULL,
  `password` varchar(10) NOT NULL,
  PRIMARY KEY  (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=2 ;

--
-- Dumping data for table `user`
--

INSERT INTO `user` (`id`, `name`, `gender`, `address`, `email`, `pnumber`, `uname`, `password`) VALUES
(1, 'sundar', 'male', 'trichy', 'sundarv06@gmail.com', '7904461600', 'sundar', 'sundar');

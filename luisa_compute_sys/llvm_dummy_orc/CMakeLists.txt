cmake_minimum_required(VERSION 3.15)
project(dummy_orc_eh)

add_library(dummy_orc_eh STATIC dummy_orc_eh.cc)
set_property(TARGET dummy_orc_eh PROPERTY POSITION_INDEPENDENT_CODE ON)

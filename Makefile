CXX := clang++
CXX_FLAGS := -O0 -g

SRC_DIR := src
INCLUDE_DIR := include
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
BIN_DIR := $(BUILD_DIR)/bin
LIB_DIR := $(BUILD_DIR)/lib


INCLUDES := -I$(INCLUDE_DIR)

SNN_MOTION_MAIN_TARGET := $(BIN_DIR)/snn-motion-main
SNN_MOTION_MAIN_SRC := $(SRC_DIR)/snn-motion-main.cpp

SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC_FILES))
STATIC_LIB := $(LIB_DIR)/libshared.a

all: $(SNN_MOTION_MAIN_TARGET)

# Build the static library
$(STATIC_LIB): $(OBJ_FILES) | $(LIB_DIR)
	@echo "Creating static library $@"
	$(AR) rcs $@ $^

# Compile the main program
$(SNN_MOTION_MAIN_TARGET): $(SNN_MOTION_MAIN_SRC) $(STATIC_LIB) | $(BIN_DIR)
	@echo "Building target $@"
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -o $@ $(SNN_MOTION_MAIN_SRC) $(STATIC_LIB)

# Compile object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@mkdir -p $(@D)
	@echo "Compiling $< -> $@"
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Ensure directories exist
$(OBJ_DIR):
	@mkdir -p $@

$(BIN_DIR):
	@mkdir -p $@

$(LIB_DIR):
	@mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)
	@echo "Build directory cleaned."

.PHONY: all clean

# Define the C compiler to use
CC = gcc

# Define any compile-time flags
CFLAGS = -Wall

# Define any libraries to link to the executable
LIBS = -lm

# Define the C source files
SRCS = ml.c mnist.c loss.c activation.c layers.c

# Define the C object files
OBJS = $(SRCS:.c=.o)

# Define the executable file
EXEC = ml

# Define arguments for the executable
ARGS = $(ARG)

.PHONY: depend clean

all: $(EXEC)

$(EXEC): clean $(OBJS)
	$(CC) $(CFLAGS) -o $(EXEC) $(OBJS) $(LIBS) && echo "Executable: ./${EXEC}"

run: $(EXEC)
	clear && ./$(EXEC) $(ARGS)

memcheck: $(EXEC)
	clear && valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose ./$(EXEC) $(ARGS)

clean: 
	rm -f $(EXEC) $(OBJS)


depend: $(SRCS)
	makedepend $^


ifndef VERBOSE
.SILENT:
endif

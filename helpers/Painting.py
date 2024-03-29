import pygame

class Painting:
    """A class that combines the LineBuilder and ColorBuilder classes to create the
    final Mondrian painting
    """
    def __init__(
        self, 
        line_builder, 
        color_builder,
        line_width=8
    ):
        self.line_builder = line_builder
        self.color_builder = color_builder
        self.height = line_builder.height
        self.width = line_builder.width
        self.line_width = line_width
        
        self.black = pygame.Color(*color_builder.black)
        self.white = pygame.Color(*color_builder.white)
        self.primary_color = pygame.Color(*color_builder.primary_color)

        # To be set later
        self.surface = None


    def setup_surface(self):
        """Create our canvas"""
        surface = pygame.Surface((self.width, self.height))
        background_color = self.white
        surface.fill(background_color)
        self.surface = surface


    def draw_lines(self):
        """Draw all the segments from a LineBuilder class"""
        segments = self.line_builder.segments
        for seg in segments['x'] + segments['y']:
            pygame.draw.line(self.surface, self.black, seg[0], seg[1], self.line_width)


    def draw_box(self):
        """Draw the primary color box from a ColorBuilder class"""
        color_builder = self.color_builder
        color_box = color_builder.get_color_box(self.line_builder.segments) if color_builder.primary_color_box is None else color_builder.primary_color_box
        pygame.draw.rect(self.surface, color_builder.primary_color, color_builder.primary_color_box)


    def draw_border(self):
        """Draw a clean border around our canvas"""
        # pygame has some weird border issues
        # This needs to be fixed in the longterm, but hardcoding works for now
        height_border = self.height - 2
        width_border = self.width - 3
        border_line_width = self.line_width + 2

        pygame.draw.line(self.surface, self.black, [0, 0], [0, height_border], border_line_width)
        pygame.draw.line(self.surface, self.black, [width_border, 0], [width_border, height_border], border_line_width)
        pygame.draw.line(self.surface, self.black, [0, height_border], [width_border, height_border], border_line_width)
        pygame.draw.line(self.surface, self.black, [0, 0], [width_border, 0], border_line_width)


    def create(self):
        """Usher the painting through the pipeline"""
        self.setup_surface()
        self.draw_box()
        self.draw_lines()
        self.draw_border()


    def save(self, filename):
        """Save the surface to the given filename"""
        pygame.image.save(self.surface, filename)
